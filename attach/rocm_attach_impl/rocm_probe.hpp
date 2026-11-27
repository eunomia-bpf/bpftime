// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "base_attach_impl.hpp"
#include <ebpf_inst.h>
namespace bpftime::attach
{
inline constexpr unsigned rocm_test_println_id = 506;
struct rocm_map_binding {
	int fd;
	MapBasicInfo info{};
	void *segment_base;
	size_t segment_size;
	// Captures a retained Array implementation, not a reusable fd. Lookup
	// returns its host address; the channel translates it for the GPU.
	std::function<uint64_t(unsigned, uint32_t, const void *, uint64_t)>
		invoke;
};
struct rocm_probe {
	int id;
	std::string kernel;
	bool at_return;
	std::string entry;
	std::string ir;
	std::vector<rocm_map_binding> maps;
};
struct rocm_probe_snapshot {
	uint64_t generation;
	std::vector<std::shared_ptr<const rocm_probe>> probes;
};
rocm_probe_snapshot snapshot_rocm_probes();
bool has_rocm_probes() noexcept;
std::string compile_rocm_ebpf(const std::vector<ebpf_inst> &,
			      const std::string &entry,
			      const std::vector<rocm_map_binding> &maps = {});
} // namespace bpftime::attach
