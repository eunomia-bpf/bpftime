#pragma once

#include "ptxpass/core.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <ebpf_inst.h>

namespace bpftime::attach
{

// Run a single pass executable with JSON stdin/stdout.
// Input JSON fields:
//  - full_ptx
//  - to_patch_kernel
//  - global_ebpf_map_info_symbol
//  - ebpf_communication_data_symbol
// Output JSON fields:
//  - output_ptx (may be empty or omitted to represent no-change)
std::optional<std::string> run_pass_executable_json(
	const std::string &exec, const std::string &full_ptx,
	const std::string &to_patch_kernel, const std::string &map_sym,
	const std::string &const_sym, const std::vector<ebpf_inst> &ebpf_insts);

std::optional<ptxpass::pass_config::PassConfig>
get_pass_config_from_executable(const std::filesystem::path &path);

} // namespace bpftime::attach
