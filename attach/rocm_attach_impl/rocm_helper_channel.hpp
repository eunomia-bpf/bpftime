// SPDX-License-Identifier: MIT
#pragma once
#include "rocm_probe.hpp"
#include "rocm_channel_abi.hpp"
#include <luthier/HSA/ApiTable.h>
#include <array>
#include <thread>

namespace bpftime::attach
{
// Owned by the instrumentation tool, and reclaimed only after its dispatches
// complete. Slots are append-only while a generation is in use.
class rocm_helper_channel {
	luthier::hsa::ApiTableContainer<AmdExtTable> amd;
	rocm_channel_data *data = nullptr;
	void *host_base = nullptr, *device_base = nullptr;
	size_t segment_size = 0;
	std::array<rocm_map_binding, rocm_max_maps> maps{};
	std::thread worker;
	void serve() noexcept;

    public:
	rocm_helper_channel(
		const luthier::hsa::ApiTableContainer<CoreApiTable> &core,
		const luthier::hsa::ApiTableContainer<AmdExtTable> &amd,
		hsa_agent_t agent, const rocm_map_binding &first);
	~rocm_helper_channel();
	rocm_helper_channel(const rocm_helper_channel &) = delete;
	rocm_helper_channel &operator=(const rocm_helper_channel &) = delete;
	uint64_t bind(const rocm_map_binding &map);
	uint64_t address() const
	{
		return reinterpret_cast<uint64_t>(data);
	}
	bool active() const
	{
		return !__atomic_load_n(&data->stopped, __ATOMIC_ACQUIRE);
	}
};
} // namespace bpftime::attach
