#pragma once

#include "bpftime-verifier.hpp"
#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace bpftime::verifier::gpu
{

std::optional<std::string> verify_gpu_program(
	const void *raw_instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors = {});

} // namespace bpftime::verifier::gpu
