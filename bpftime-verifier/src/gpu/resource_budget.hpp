#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ebpf_verifier_stats_t;
struct ebpf_inst;

namespace bpftime::verifier::gpu
{

struct GpuResourceBudget {
	uint32_t max_instructions = 4096;
	uint32_t max_helper_calls = 64;
	uint32_t max_memory_ops = 256;
	uint32_t max_map_lookups = 32;
	uint32_t max_map_updates = 16;
};

struct ResourceBudgetResult {
	bool passed = true;
	std::string error_message;
	uint32_t instruction_count = 0;
	uint32_t helper_call_count = 0;
	uint32_t memory_op_count = 0;
	uint32_t map_lookup_count = 0;
	uint32_t map_update_count = 0;
};

GpuResourceBudget get_default_budget(const std::string &section_name);

ResourceBudgetResult
check_resource_budget(const ebpf_inst *instructions, size_t num_instructions,
		       const GpuResourceBudget &budget,
		       const ebpf_verifier_stats_t *stats = nullptr);

} // namespace bpftime::verifier::gpu
