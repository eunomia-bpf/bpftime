// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <ebpf_inst.h>
namespace bpftime::attach
{
inline constexpr unsigned rocm_test_println_id = 506;
struct rocm_probe {
	int id;
	std::string kernel;
	bool at_return;
	std::string entry;
	std::string ir;
};
struct rocm_probe_snapshot {
	uint64_t generation;
	std::vector<std::shared_ptr<const rocm_probe>> probes;
};
rocm_probe_snapshot snapshot_rocm_probes();
bool has_rocm_probes() noexcept;
std::string compile_rocm_ebpf(const std::vector<ebpf_inst> &,
			      const std::string &entry);
} // namespace bpftime::attach
