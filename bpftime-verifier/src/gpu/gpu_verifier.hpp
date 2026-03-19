#pragma once

#include "bpftime-verifier.hpp"
#include "resource_budget.hpp"
#include "uniformity_analysis.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

struct ebpf_inst;

namespace bpftime::verifier::gpu
{

enum class GpuVerificationMode {
	SIMT_AWARE,
	PREVAIL_ONLY,
};

struct GpuVerifierConfig {
	std::optional<GpuResourceBudget> budget;
	bool strict_uniformity = true;
	bool allow_membar = false;
	bool skip_prevail = false;
	std::map<int, BpftimeMapDescriptor> map_descriptors;
	GpuVerificationMode mode = GpuVerificationMode::SIMT_AWARE;
};

struct GpuVerifyResult {
	bool passed = false;
	std::string error_message;
	double total_time_us = 0.0;
	double prevail_time_us = 0.0;
	double simt_time_us = 0.0;
	uint32_t instruction_count = 0;
	uint32_t helper_call_count = 0;
	uint32_t memory_op_count = 0;
	uint32_t map_lookup_count = 0;
	uint32_t map_update_count = 0;
	uint32_t varying_branch_count = 0;
	uint32_t prohibited_helper_count = 0;
	std::vector<Uniformity> final_reg_uniformity;
};

GpuVerifyResult verify_gpu_program(const ebpf_inst *instructions,
				   size_t num_instructions,
				   const std::string &section_name,
				   const GpuVerifierConfig &config = {});

GpuVerifyResult verify_gpu_program(const uint64_t *raw_inst,
				   size_t num_instructions,
				   const std::string &section_name,
				   const GpuVerifierConfig &config = {});

} // namespace bpftime::verifier::gpu
