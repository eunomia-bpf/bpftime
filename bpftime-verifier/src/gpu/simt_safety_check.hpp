#pragma once

#include "bpftime-verifier.hpp"
#include "uniformity_analysis.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ebpf_inst;

namespace bpftime::verifier::gpu
{

struct SimtCheckOptions {
	bool strict_uniformity = true;
	bool allow_prohibited_sync = false;
	std::map<int, BpftimeMapDescriptor> map_descriptors;
};

struct SimtSafetyError {
	size_t instruction_index = 0;
	std::string check_name;
	std::string message;
};

struct SimtSafetyResult {
	bool passed = true;
	uint32_t varying_branch_count = 0;
	uint32_t prohibited_helper_count = 0;
	uint32_t varying_atomic_count = 0;
	uint32_t varying_map_key_count = 0;
	std::vector<SimtSafetyError> errors;

	std::string summary() const;
};

SimtSafetyResult check_simt_safety(const ebpf_inst *instructions,
				     size_t num_instructions,
				     const UniformityAnalysisResult &uniformity,
				     const SimtCheckOptions &options = {});

} // namespace bpftime::verifier::gpu
