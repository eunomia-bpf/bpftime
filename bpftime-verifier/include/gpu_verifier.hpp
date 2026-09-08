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

// Automatic warp-execution admission. Runs the full GPU verification
// pipeline and the warp-policy eligibility rules. Returns std::nullopt when
// the program may be lowered to once-per-active-warp leader execution at its
// hook sites; otherwise returns the rejection reason (verification failure
// or transformation ineligibility). This is only consulted to select an
// execution mode; it does not change which programs are admitted for load.
std::optional<std::string> evaluate_warp_execution_eligibility(
	const void *raw_instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, BpftimeMapDescriptor> &map_descriptors = {});

} // namespace bpftime::verifier::gpu
