#include "gpu_verifier.hpp"

#include "asm_unmarshal.hpp"
#include "crab_verifier.hpp"
#include "ebpf_vm_isa.hpp"
#include "gpu_platform.hpp"
#include "platform-impl.hpp"
#include "simt_safety_check.hpp"
#include "spec_type_descriptors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <variant>

namespace
{

ebpf_verifier_options_t gpu_verifier_options = {
	.check_termination = true,
	.assume_assertions = false,
	.print_invariants = true,
	.print_failures = true,
	.no_simplify = true,
	.mock_map_fds = false,
	.strict = false,
	.print_line_info = true,
	.allow_division_by_zero = false,
	.setup_constraints = false,
	.dump_btf_types_json = false,
};

constexpr std::array<int32_t, 6> PREVAIL_STANDARD_HELPERS = {
	1, 2, 3, 6, 14, 25,
};

constexpr std::array<int32_t, 11> PREVAIL_GPU_HELPERS = {
	501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511,
};

bpftime::verifier::BpftimeHelperProrotype
to_bpftime_helper_prototype(const EbpfHelperPrototype &prototype)
{
	bpftime::verifier::BpftimeHelperProrotype result{};
	result.name = prototype.name;
	result.return_type =
		static_cast<bpftime::verifier::bpftime_return_type_t>(
			static_cast<int>(prototype.return_type));
	for (size_t i = 0; i < 5; ++i) {
		result.argument_type[i] =
			static_cast<bpftime::verifier::bpftime_argument_type_t>(
				static_cast<int>(prototype.argument_type[i]));
	}
	return result;
}

std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype>
get_non_kernel_helper_overrides()
{
	std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype> helpers;
	for (const auto &[helper_id, prototype] : bpftime::non_kernel_helpers) {
		helpers.emplace(helper_id,
				to_bpftime_helper_prototype(prototype));
	}
	return helpers;
}

bpftime::verifier::BpftimeHelperProrotype
make_prevail_gpu_helper_override(int32_t helper_id)
{
	const auto *helper = bpftime::find_gpu_helper_prototype(helper_id);
	if (helper == nullptr) {
		throw std::runtime_error("Missing GPU helper prototype: " +
					 std::to_string(helper_id));
	}

	bpftime::verifier::BpftimeHelperProrotype prototype{};
	prototype.name = helper->name;
	prototype.return_type =
		static_cast<bpftime::verifier::bpftime_return_type_t>(
			static_cast<int>(helper->return_type));

	size_t arity = 0;
	for (; arity < helper->prevail_argument_types.size(); ++arity) {
		if (helper->prevail_argument_types[arity] ==
		    EBPF_ARGUMENT_TYPE_DONTCARE) {
			break;
		}
	}

	for (size_t i = 0; i < 5; ++i) {
		prototype.argument_type[i] =
			i < arity ?
				bpftime::verifier::EBPF_ARGUMENT_TYPE_ANYTHING :
				bpftime::verifier::EBPF_ARGUMENT_TYPE_DONTCARE;
	}

	return prototype;
}

std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype>
make_prevail_gpu_helper_overrides()
{
	auto helpers = get_non_kernel_helper_overrides();
	for (const auto helper_id : PREVAIL_GPU_HELPERS) {
		helpers[helper_id] =
			make_prevail_gpu_helper_override(helper_id);
	}
	return helpers;
}

bool is_call(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_CALL;
}

bool is_jump_with_offset(const ebpf_inst &instruction)
{
	const uint8_t instruction_class = instruction.opcode & INST_CLS_MASK;
	return (instruction_class == INST_CLS_JMP ||
		instruction_class == INST_CLS_JMP32) &&
	       instruction.opcode != INST_OP_CALL &&
	       instruction.opcode != INST_OP_EXIT;
}

size_t prevail_outparam_store_count(int32_t helper_id)
{
	const auto *helper = bpftime::find_gpu_helper_prototype(helper_id);
	if (helper == nullptr) {
		return 0;
	}

	return static_cast<size_t>(std::count(
		helper->semantic_argument_types.begin(),
		helper->semantic_argument_types.end(),
		bpftime::GpuHelperArgumentSemantics::PTR_TO_U64_OUT));
}

ebpf_inst make_stdw_imm(uint8_t dst_reg, int16_t offset, int32_t imm)
{
	ebpf_inst instruction{};
	instruction.opcode = INST_CLS_ST | (INST_MEM << 5) | INST_SIZE_DW;
	instruction.dst = dst_reg;
	instruction.offset = offset;
	instruction.imm = imm;
	return instruction;
}

std::vector<ebpf_inst>
build_prevail_shadow_program(const ebpf_inst *instructions,
			     size_t num_instructions)
{
	std::vector<size_t> prelude_counts(num_instructions, 0);
	size_t total_inserted_instructions = 0;

	for (size_t pc = 0; pc < num_instructions; ++pc) {
		if (!is_call(instructions[pc])) {
			continue;
		}
		prelude_counts[pc] =
			prevail_outparam_store_count(instructions[pc].imm);
		total_inserted_instructions += prelude_counts[pc];
	}

	if (total_inserted_instructions == 0) {
		return std::vector<ebpf_inst>(instructions,
					      instructions + num_instructions);
	}

	std::vector<size_t> entry_index(num_instructions, 0);
	std::vector<size_t> instruction_index(num_instructions, 0);
	size_t cursor = 0;
	for (size_t pc = 0; pc < num_instructions; ++pc) {
		entry_index[pc] = cursor;
		cursor += prelude_counts[pc];
		instruction_index[pc] = cursor;
		++cursor;
	}
	const size_t shadow_size = cursor;

	std::vector<ebpf_inst> shadow_program;
	shadow_program.reserve(shadow_size);

	for (size_t pc = 0; pc < num_instructions; ++pc) {
		if (prelude_counts[pc] > 0) {
			const auto *helper = bpftime::find_gpu_helper_prototype(
				instructions[pc].imm);
			if (helper == nullptr) {
				throw std::runtime_error(
					"Missing GPU helper prototype: " +
					std::to_string(instructions[pc].imm));
			}
			for (size_t i = 0;
			     i < helper->semantic_argument_types.size(); ++i) {
				if (helper->semantic_argument_types[i] !=
				    bpftime::GpuHelperArgumentSemantics::
					    PTR_TO_U64_OUT) {
					continue;
				}
				shadow_program.push_back(make_stdw_imm(
					static_cast<uint8_t>(i + 1), 0, 0));
			}
		}

		ebpf_inst instruction = instructions[pc];
		if (is_jump_with_offset(instruction)) {
			const int64_t original_target =
				static_cast<int64_t>(pc) + 1 +
				static_cast<int64_t>(instruction.offset);
			if (original_target < 0 ||
			    original_target >
				    static_cast<int64_t>(num_instructions)) {
				throw std::runtime_error(
					"Invalid jump target while building PREVAIL shadow program");
			}

			const size_t shadow_target =
				original_target == static_cast<int64_t>(
							   num_instructions) ?
					shadow_size :
					entry_index[static_cast<size_t>(
						original_target)];
			const int64_t shadow_offset =
				static_cast<int64_t>(shadow_target) -
				static_cast<int64_t>(instruction_index[pc]) - 1;
			if (shadow_offset <
				    std::numeric_limits<int16_t>::min() ||
			    shadow_offset >
				    std::numeric_limits<int16_t>::max()) {
				throw std::runtime_error(
					"PREVAIL shadow jump offset overflow");
			}
			instruction.offset =
				static_cast<int16_t>(shadow_offset);
		}

		shadow_program.push_back(instruction);
	}

	return shadow_program;
}

std::vector<EbpfMapDescriptor> to_prevail_map_descriptors(
	const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps)
{
	std::vector<EbpfMapDescriptor> descriptors;
	descriptors.reserve(maps.size());
	for (const auto &[fd, map] : maps) {
		(void)fd;
		descriptors.push_back(EbpfMapDescriptor{
			.original_fd = map.original_fd,
			.type = map.type,
			.key_size = map.key_size,
			.value_size = map.value_size,
			.max_entries = map.max_entries,
			.inner_map_fd = map.inner_map_fd,
		});
	}
	return descriptors;
}

std::map<int, bpftime::verifier::BpftimeMapDescriptor>
effective_map_descriptors(
	const bpftime::verifier::gpu::GpuVerifierConfig &config)
{
	if (!config.map_descriptors.empty()) {
		return config.map_descriptors;
	}
	return bpftime::verifier::get_map_descriptors();
}

raw_program make_raw_program(const ebpf_inst *instructions, size_t count,
			     const std::string &section_name,
			     const std::vector<EbpfMapDescriptor> &maps)
{
	raw_program program;
	program.filename = "BPFTIME_GPU_VERIFIER";
	program.section = section_name;
	program.prog.assign(instructions, instructions + count);
	program.info = {
		.platform = &bpftime::gpu_platform_spec,
		.map_descriptors = maps,
		.type = bpftime::gpu_platform_spec.get_program_type(
			section_name, ""),
	};
	return program;
}

std::vector<bpftime::verifier::gpu::Uniformity>
final_uniformity(const bpftime::verifier::gpu::UniformityAnalysisResult &result)
{
	std::vector<bpftime::verifier::gpu::Uniformity> regs;
	if (result.states.empty()) {
		return regs;
	}
	regs.assign(result.states.back().regs.begin(),
		    result.states.back().regs.end());
	return regs;
}

class ScopedPrevailGpuRegistration {
    public:
	explicit ScopedPrevailGpuRegistration(
		const std::map<int, bpftime::verifier::BpftimeMapDescriptor>
			&maps)
		: previous_available_helpers_(bpftime::usable_helpers.begin(),
					      bpftime::usable_helpers.end()),
		  previous_non_kernel_helpers_(
			  get_non_kernel_helper_overrides()),
		  previous_maps_(bpftime::verifier::get_map_descriptors())
	{
		std::set<int32_t> helpers(bpftime::usable_helpers.begin(),
					  bpftime::usable_helpers.end());
		helpers.insert(PREVAIL_STANDARD_HELPERS.begin(),
			       PREVAIL_STANDARD_HELPERS.end());
		helpers.insert(PREVAIL_GPU_HELPERS.begin(),
			       PREVAIL_GPU_HELPERS.end());
		bpftime::verifier::set_available_helpers(
			std::vector<int32_t>(helpers.begin(), helpers.end()));
		bpftime::verifier::set_non_kernel_helpers(
			make_prevail_gpu_helper_overrides());
		bpftime::verifier::set_map_descriptors(maps);
	}

	~ScopedPrevailGpuRegistration()
	{
		bpftime::verifier::set_available_helpers(
			previous_available_helpers_);
		bpftime::verifier::set_non_kernel_helpers(
			previous_non_kernel_helpers_);
		bpftime::verifier::set_map_descriptors(previous_maps_);
	}

    private:
	std::vector<int32_t> previous_available_helpers_;
	std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype>
		previous_non_kernel_helpers_;
	std::map<int, bpftime::verifier::BpftimeMapDescriptor> previous_maps_;
};

struct PrevailAttemptResult {
	bool attempted = false;
	bool passed = false;
	bool used_shadow_program = false;
	std::string message;
	ebpf_verifier_stats_t stats{};
};

template <typename Clock>
double elapsed_us(typename Clock::time_point start,
		  typename Clock::time_point end)
{
	return std::chrono::duration<double, std::micro>(end - start).count();
}

template <typename Clock>
void finish_gpu_stage(bpftime::verifier::gpu::GpuVerifyResult &result,
		      typename Clock::time_point total_start,
		      typename Clock::time_point simt_start,
		      typename Clock::time_point end)
{
	result.simt_time_us = elapsed_us<Clock>(simt_start, end);
	result.total_time_us = elapsed_us<Clock>(total_start, end);
}

template <typename Clock>
PrevailAttemptResult
run_prevail(const ebpf_inst *instructions, size_t num_instructions,
	    const std::string &section_name,
	    const std::vector<EbpfMapDescriptor> &prevail_maps,
	    const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps)
{
	PrevailAttemptResult result;
	result.attempted = true;
	result.stats = ebpf_verifier_stats_t{};

	try {
		ScopedPrevailGpuRegistration scoped_registration(maps);
		const auto prevail_instructions = build_prevail_shadow_program(
			instructions, num_instructions);
		result.used_shadow_program =
			prevail_instructions.size() != num_instructions;
		auto program = make_raw_program(prevail_instructions.data(),
						prevail_instructions.size(),
						section_name, prevail_maps);

		std::vector<std::vector<std::string>> notes;
		auto unmarshal_result = unmarshal(program, notes);
		if (std::holds_alternative<std::string>(unmarshal_result)) {
			result.message =
				std::get<std::string>(unmarshal_result);
			return result;
		}

		std::ostringstream prevail_message;
		result.passed = ebpf_verify_program(
			prevail_message,
			std::get<InstructionSeq>(unmarshal_result),
			program.info, &gpu_verifier_options, &result.stats);
		if (!result.passed) {
			result.message = prevail_message.str();
		}
	} catch (const std::exception &ex) {
		result.message = ex.what();
	} catch (...) {
		result.message = "unknown PREVAIL exception";
	}
	return result;
}

} // namespace

namespace bpftime::verifier::gpu
{

GpuVerifyResult verify_gpu_program(const ebpf_inst *instructions,
				   size_t num_instructions,
				   const std::string &section_name,
				   const GpuVerifierConfig &config)
{
	using Clock = std::chrono::steady_clock;
	const auto total_start = Clock::now();

	GpuVerifyResult result;
	if (instructions == nullptr) {
		result.error_message = "null instruction stream";
		result.total_time_us = 0.0;
		return result;
	}
	if (num_instructions == 0) {
		result.passed = true;
		result.total_time_us = 0.0;
		return result;
	}

	const auto maps = effective_map_descriptors(config);
	const auto run_prevail_now = !config.skip_prevail;
	if (config.mode == GpuVerificationMode::PREVAIL_ONLY &&
	    config.skip_prevail) {
		result.error_message =
			"PREVAIL_ONLY mode requires skip_prevail = false";
		result.total_time_us = 0.0;
		return result;
	}

	if (config.mode == GpuVerificationMode::PREVAIL_ONLY) {
		const auto prevail_start = Clock::now();
		const auto prevail = run_prevail<Clock>(
			instructions, num_instructions, section_name,
			to_prevail_map_descriptors(maps), maps);
		const auto prevail_end = Clock::now();
		result.prevail_time_us =
			elapsed_us<Clock>(prevail_start, prevail_end);
		result.total_time_us = result.prevail_time_us;
		result.passed = prevail.passed;
		result.error_message = prevail.message;
		result.simt_time_us = 0.0;
		return result;
	}

	const ebpf_verifier_stats_t *budget_stats = nullptr;
	ebpf_verifier_stats_t prevail_stats{};
	if (run_prevail_now) {
		const auto prevail_start = Clock::now();
		const auto prevail = run_prevail<Clock>(
			instructions, num_instructions, section_name,
			to_prevail_map_descriptors(maps), maps);
		const auto prevail_end = Clock::now();
		result.prevail_time_us =
			elapsed_us<Clock>(prevail_start, prevail_end);
		if (!prevail.passed) {
			result.error_message = prevail.message;
			result.total_time_us =
				elapsed_us<Clock>(total_start, prevail_end);
			return result;
		}
		prevail_stats = prevail.stats;
		if (!prevail.used_shadow_program) {
			budget_stats = &prevail_stats;
		}
	}

	const auto simt_start = Clock::now();
	const auto uniformity =
		analyze_uniformity(instructions, num_instructions, maps);
	result.final_reg_uniformity = final_uniformity(uniformity);
	if (!uniformity.success) {
		result.error_message = uniformity.error_message;
		const auto total_end = Clock::now();
		finish_gpu_stage<Clock>(result, total_start, simt_start,
					total_end);
		return result;
	}

	SimtCheckOptions simt_options{
		.strict_uniformity = config.strict_uniformity,
		.allow_prohibited_sync = config.allow_membar,
		.map_descriptors = maps,
	};
	const auto simt = check_simt_safety(instructions, num_instructions,
					    uniformity, simt_options);
	result.varying_branch_count = simt.varying_branch_count;
	result.prohibited_helper_count = simt.prohibited_helper_count;
	if (!simt.passed) {
		result.error_message = simt.summary();
		const auto total_end = Clock::now();
		finish_gpu_stage<Clock>(result, total_start, simt_start,
					total_end);
		return result;
	}

	const auto budget =
		config.budget.value_or(get_default_budget(section_name));
	const auto budget_result = check_resource_budget(
		instructions, num_instructions, budget, budget_stats);
	result.instruction_count = budget_result.instruction_count;
	result.helper_call_count = budget_result.helper_call_count;
	result.memory_op_count = budget_result.memory_op_count;
	result.map_lookup_count = budget_result.map_lookup_count;
	result.map_update_count = budget_result.map_update_count;
	if (!budget_result.passed) {
		result.error_message = budget_result.error_message;
		const auto total_end = Clock::now();
		finish_gpu_stage<Clock>(result, total_start, simt_start,
					total_end);
		return result;
	}

	result.passed = true;
	const auto total_end = Clock::now();
	finish_gpu_stage<Clock>(result, total_start, simt_start, total_end);
	return result;
}

GpuVerifyResult verify_gpu_program(const uint64_t *raw_inst,
				   size_t num_instructions,
				   const std::string &section_name,
				   const GpuVerifierConfig &config)
{
	std::vector<ebpf_inst> instructions(num_instructions);
	for (size_t i = 0; i < num_instructions; ++i) {
		memcpy(&instructions[i], &raw_inst[i], sizeof(ebpf_inst));
	}
	return verify_gpu_program(instructions.data(), instructions.size(),
				  section_name, config);
}

} // namespace bpftime::verifier::gpu
