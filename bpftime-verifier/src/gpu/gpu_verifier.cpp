#include "gpu_verifier.hpp"

#include "asm_unmarshal.hpp"
#include "crab_verifier.hpp"
#include "ebpf_vm_isa.hpp"
#include "gpu_platform.hpp"
#include "platform-impl.hpp"
#include "simt_safety_check.hpp"
#include "spec_type_descriptors.hpp"
#include "uniformity_analysis.hpp"

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
	.print_invariants = false,
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
		if (i >= arity) {
			prototype.argument_type[i] =
				bpftime::verifier::EBPF_ARGUMENT_TYPE_DONTCARE;
		} else if (helper_id == 501) {
			prototype.argument_type[i] = static_cast<
				bpftime::verifier::bpftime_argument_type_t>(
				static_cast<int>(
					helper->prevail_argument_types[i]));
		} else {
			prototype.argument_type[i] =
				bpftime::verifier::EBPF_ARGUMENT_TYPE_ANYTHING;
		}
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
	const std::map<int, bpftime::verifier::BpftimeMapDescriptor>
		&map_descriptors)
{
	if (!map_descriptors.empty()) {
		return map_descriptors;
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
		std::set<int32_t> helpers(PREVAIL_STANDARD_HELPERS.begin(),
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
	bool passed = false;
	std::string message;
};

template <typename Clock>
double elapsed_us(typename Clock::time_point start,
		  typename Clock::time_point end)
{
	return std::chrono::duration<double, std::micro>(end - start).count();
}

PrevailAttemptResult
run_prevail(const ebpf_inst *instructions, size_t num_instructions,
	    const std::string &section_name,
	    const std::vector<EbpfMapDescriptor> &prevail_maps,
	    const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps)
{
	PrevailAttemptResult result;
	ebpf_verifier_stats_t stats{};

	try {
		ScopedPrevailGpuRegistration scoped_registration(maps);
		const auto prevail_instructions = build_prevail_shadow_program(
			instructions, num_instructions);
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
			program.info, &gpu_verifier_options, &stats);
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

static GpuVerifyResult verify_gpu_instructions(
	const ebpf_inst *instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors)
{
	using Clock = std::chrono::steady_clock;
	const auto total_start = Clock::now();

	GpuVerifyResult result;
	auto finish = [&]() {
		result.total_time_us =
			elapsed_us<Clock>(total_start, Clock::now());
		return result;
	};
	if (instructions == nullptr) {
		result.error_message = "null instruction stream";
		return finish();
	}
	if (num_instructions == 0) {
		result.passed = true;
		return finish();
	}

	const auto maps = effective_map_descriptors(map_descriptors);
	for (const auto &[fd, map] : maps) {
		(void)fd;
		if (map.type >= 1500 && map.type < 1600 &&
		    !bpftime::try_get_gpu_map_type(map.type).has_value()) {
			result.error_message = "unsupported GPU map type " +
					       std::to_string(map.type);
			return finish();
		}
	}
	const auto prevail =
		run_prevail(instructions, num_instructions, section_name,
			    to_prevail_map_descriptors(maps), maps);
	if (!prevail.passed) {
		result.error_message = prevail.message;
		return finish();
	}
	const auto uniformity =
		analyze_uniformity(instructions, num_instructions, maps);
	if (!uniformity.success) {
		result.error_message = uniformity.error_message;
		return finish();
	}

	const auto simt = check_simt_safety(instructions, num_instructions,
					    uniformity, maps);
	if (!simt.passed) {
		result.error_message = simt.summary();
		return finish();
	}

	result.passed = true;
	return finish();
}

GpuVerifyResult
verify_gpu_program(const void *raw_instructions, size_t num_instructions,
		   const std::string &section_name,
		   const std::map<int, BpftimeMapDescriptor> &map_descriptors)
{
	static_assert(sizeof(ebpf_inst) == sizeof(uint64_t));
	if (raw_instructions == nullptr) {
		return verify_gpu_instructions(nullptr, num_instructions,
					       section_name, map_descriptors);
	}

	const auto *bytes = static_cast<const std::byte *>(raw_instructions);
	std::vector<ebpf_inst> instructions(num_instructions);
	for (size_t i = 0; i < num_instructions; ++i) {
		std::memcpy(&instructions[i], bytes + i * sizeof(ebpf_inst),
			    sizeof(ebpf_inst));
	}
	return verify_gpu_instructions(instructions.data(), instructions.size(),
				       section_name, map_descriptors);
}

} // namespace bpftime::verifier::gpu
