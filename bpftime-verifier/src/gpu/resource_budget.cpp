#include "resource_budget.hpp"

#include "config.hpp"
#include "gpu_platform.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string_view>

namespace
{

using bpftime::GpuHelperBehavior;
using bpftime::verifier::gpu::GpuResourceBudget;
using bpftime::verifier::gpu::ResourceBudgetResult;

bool is_lddw_continuation(const ebpf_inst *instructions, size_t pc)
{
	return pc > 0 && instructions[pc - 1].opcode == INST_OP_LDDW_IMM;
}

bool is_call(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_CALL;
}

bool is_memory_op(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	return cls == INST_CLS_LDX || cls == INST_CLS_ST ||
	       cls == INST_CLS_STX;
}

uint32_t saturating_cast(uint64_t value)
{
	return value > std::numeric_limits<uint32_t>::max() ?
		       std::numeric_limits<uint32_t>::max() :
		       static_cast<uint32_t>(value);
}

uint64_t ceiling_divide(uint64_t numerator, uint64_t denominator)
{
	return denominator == 0 ? 1 : (numerator + denominator - 1) / denominator;
}

} // namespace

namespace bpftime::verifier::gpu
{

GpuResourceBudget get_default_budget(const std::string &section_name)
{
	const std::string_view section(section_name);
	if (section.find("memcapture") != std::string_view::npos) {
		return GpuResourceBudget{ 2048, 32, 128, 16, 8 };
	}
	if (section.find("directly_run") != std::string_view::npos) {
		return GpuResourceBudget{ 8192, 128, 512, 64, 32 };
	}
	if (section.find("scheduler") != std::string_view::npos) {
		return GpuResourceBudget{ 1024, 16, 64, 8, 4 };
	}
	return GpuResourceBudget{};
}

ResourceBudgetResult
check_resource_budget(const ebpf_inst *instructions, size_t num_instructions,
		       const GpuResourceBudget &budget,
		       const ebpf_verifier_stats_t *stats)
{
	ResourceBudgetResult result;
	uint64_t static_instruction_count = 0;
	uint64_t static_helper_call_count = 0;
	uint64_t static_memory_op_count = 0;
	uint64_t static_map_lookup_count = 0;
	uint64_t static_map_update_count = 0;

	for (size_t pc = 0; pc < num_instructions; ++pc) {
		if (is_lddw_continuation(instructions, pc)) {
			continue;
		}
		++static_instruction_count;

		const auto &instruction = instructions[pc];
		if (is_memory_op(instruction)) {
			++static_memory_op_count;
		}
		if (!is_call(instruction)) {
			continue;
		}

		++static_helper_call_count;
		const auto helper = bpftime::get_gpu_helper_effects(instruction.imm);
		if (helper.behavior == GpuHelperBehavior::MAP_LOOKUP) {
			++static_map_lookup_count;
		}
		if (helper.behavior == GpuHelperBehavior::MAP_UPDATE) {
			++static_map_update_count;
		}
	}

	uint64_t worst_case_instruction_count = static_instruction_count;
	if (stats != nullptr && stats->max_instruction_count > 0) {
		worst_case_instruction_count = std::max<uint64_t>(
			worst_case_instruction_count,
			static_cast<uint64_t>(stats->max_instruction_count));
	}

	// PREVAIL currently exposes only whole-program max_instruction_count.
	// Use that as the instruction budget directly, then scale helper and
	// memory-style counts by the same factor as an approximation. This keeps
	// loop accounting honest without pretending we have per-helper loop bounds.
	const uint64_t loop_multiplier =
		std::max<uint64_t>(1, ceiling_divide(worst_case_instruction_count,
						      std::max<uint64_t>(
							      1, static_instruction_count)));

	result.instruction_count = saturating_cast(worst_case_instruction_count);
	result.helper_call_count =
		saturating_cast(static_helper_call_count * loop_multiplier);
	result.memory_op_count =
		saturating_cast(static_memory_op_count * loop_multiplier);
	result.map_lookup_count =
		saturating_cast(static_map_lookup_count * loop_multiplier);
	result.map_update_count =
		saturating_cast(static_map_update_count * loop_multiplier);

	std::ostringstream error;
	bool first_error = true;
	auto append_error = [&](const char *label, uint32_t actual,
				uint32_t limit) {
		if (first_error) {
			error << "resource budget exceeded: ";
			first_error = false;
		} else {
			error << "; ";
		}
		error << label << " " << actual << " > " << limit;
	};

	if (result.instruction_count > budget.max_instructions) {
		append_error("instruction count", result.instruction_count,
			     budget.max_instructions);
	}
	if (result.helper_call_count > budget.max_helper_calls) {
		append_error("helper call count", result.helper_call_count,
			     budget.max_helper_calls);
	}
	if (result.memory_op_count > budget.max_memory_ops) {
		append_error("memory operation count", result.memory_op_count,
			     budget.max_memory_ops);
	}
	if (result.map_lookup_count > budget.max_map_lookups) {
		append_error("map lookup count", result.map_lookup_count,
			     budget.max_map_lookups);
	}
	if (result.map_update_count > budget.max_map_updates) {
		append_error("map update count", result.map_update_count,
			     budget.max_map_updates);
	}

	result.passed = first_error;
	result.error_message = error.str();
	return result;
}

} // namespace bpftime::verifier::gpu
