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
#include <cstring>
#include <exception>
#include <limits>
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

std::optional<std::string>
run_prevail(const ebpf_inst *instructions, size_t num_instructions,
	    const std::string &section_name,
	    const std::vector<EbpfMapDescriptor> &prevail_maps)
{
	ebpf_verifier_stats_t stats{};

	try {
		const auto prevail_instructions = build_prevail_shadow_program(
			instructions, num_instructions);
		auto program = make_raw_program(prevail_instructions.data(),
						prevail_instructions.size(),
						section_name, prevail_maps);

		std::vector<std::vector<std::string>> notes;
		auto unmarshal_result = unmarshal(program, notes);
		if (std::holds_alternative<std::string>(unmarshal_result)) {
			return std::get<std::string>(unmarshal_result);
		}

		std::ostringstream prevail_message;
		if (!ebpf_verify_program(
			    prevail_message,
			    std::get<InstructionSeq>(unmarshal_result),
			    program.info, &gpu_verifier_options, &stats)) {
			return prevail_message.str();
		}
	} catch (const std::exception &ex) {
		return ex.what();
	} catch (...) {
		return "unknown PREVAIL exception";
	}
	return std::nullopt;
}

} // namespace

namespace bpftime::verifier::gpu
{

static std::optional<std::string> verify_gpu_instructions(
	const ebpf_inst *instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors)
{
	if (num_instructions == 0) {
		return "empty instruction stream";
	}
	if (instructions == nullptr) {
		return "null instruction stream";
	}

	const auto &maps = map_descriptors;
	for (const auto &[fd, map] : maps) {
		(void)fd;
		if (map.type >= 1500 && map.type < 1600 &&
		    !bpftime::try_get_gpu_map_type(map.type).has_value()) {
			return "unsupported GPU map type " +
			       std::to_string(map.type);
		}
	}
	if (auto error =
		    run_prevail(instructions, num_instructions, section_name,
				to_prevail_map_descriptors(maps))) {
		return error;
	}
	const auto uniformity =
		analyze_uniformity(instructions, num_instructions, maps);
	if (!uniformity.success) {
		return uniformity.error_message;
	}

	const auto simt = check_simt_safety(instructions, num_instructions,
					    uniformity, maps);
	if (!simt.passed) {
		return simt.summary();
	}

	return std::nullopt;
}

std::optional<std::string>
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
