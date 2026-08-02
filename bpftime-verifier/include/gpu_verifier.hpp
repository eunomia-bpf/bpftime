#pragma once

#include "bpftime-verifier.hpp"
#include <cstddef>
#include <map>
#include <string>

namespace bpftime::verifier::gpu
{

struct GpuVerifyResult {
	bool passed = false;
	std::string error_message;
};

GpuVerifyResult verify_gpu_program(
	const void *raw_instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors = {});

} // namespace bpftime::verifier::gpu
