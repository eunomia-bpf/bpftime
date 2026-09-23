#pragma once

#include "bpftime-verifier.hpp"
#include "uniformity_analysis.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

struct ebpf_inst;

namespace bpftime::verifier::gpu
{

struct SimtSafetyError {
	size_t instruction_index = 0;
	std::string check_name;
	std::string message;
};

struct SimtSafetyResult {
	bool passed = true;
	std::vector<SimtSafetyError> errors;

	std::string summary() const;
};

SimtSafetyResult check_simt_safety(
	const ebpf_inst *instructions, size_t num_instructions,
	const UniformityAnalysisResult &uniformity,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors = {});

} // namespace bpftime::verifier::gpu
