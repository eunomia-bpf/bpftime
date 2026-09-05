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

// Verify against an explicit read/write context buffer. The default GPU
// verifier deliberately exposes no context; standalone AOT entries use this
// variant so PREVAIL can prove every access against `context_size`.
std::optional<std::string> verify_gpu_program_with_context(
	const void *raw_instructions, size_t num_instructions,
	const std::string &section_name, size_t context_size,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors = {});

} // namespace bpftime::verifier::gpu
