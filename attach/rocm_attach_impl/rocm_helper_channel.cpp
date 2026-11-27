// SPDX-License-Identifier: MIT
#include "rocm_helper_channel.hpp"
#include <luthier/HSA/Agent.h>
#include <luthier/HSA/MemoryPool.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <cerrno>

namespace bpftime::attach
{
namespace
{
void check(hsa_status_t status)
{
	if (status != HSA_STATUS_SUCCESS)
		throw std::runtime_error("ROCm map memory setup failed: " +
					 std::to_string(status));
}
} // namespace
rocm_helper_channel::rocm_helper_channel(
	const luthier::hsa::ApiTableContainer<CoreApiTable> &core,
	const luthier::hsa::ApiTableContainer<AmdExtTable> &api,
	hsa_agent_t agent, const rocm_map_binding &first)
	: amd(api)
{
	if (!amd.tableSupportsFunction<hsa_amd_memory_lock_to_pool>() ||
	    !amd.tableSupportsFunction<hsa_amd_memory_unlock>() ||
	    !amd.tableSupportsFunction<hsa_amd_memory_pool_allocate>() ||
	    !amd.tableSupportsFunction<hsa_amd_memory_pool_free>() ||
	    !amd.tableSupportsFunction<hsa_amd_agents_allow_access>() ||
	    !amd.tableSupportsFunction<hsa_amd_agent_iterate_memory_pools>() ||
	    !amd.tableSupportsFunction<hsa_amd_memory_pool_get_info>() ||
	    !amd.tableSupportsFunction<hsa_amd_agent_memory_pool_get_info>())
		throw std::runtime_error(
			"ROCm map memory APIs are unavailable");
	llvm::SmallVector<hsa_agent_t> cpus;
	if (auto error = luthier::hsa::getAllAgentsWithDeviceType<
		    HSA_DEVICE_TYPE_CPU>(core, cpus))
		throw std::runtime_error(llvm::toString(std::move(error)));
	hsa_amd_memory_pool_t pool{};
	for (auto cpu : cpus) {
		auto error = luthier::hsa::agentIterateMemoryPools(
			amd, cpu,
			[&](hsa_amd_memory_pool_t candidate) -> llvm::Error {
				try {
					hsa_amd_segment_t segment{};
					uint32_t flags = 0, hops = 0;
					bool alloc = false;
					hsa_amd_memory_pool_access_t access{};
					if (amd.callFunction<
						    hsa_amd_memory_pool_get_info>(
						    candidate,
						    HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
						    &segment) !=
						    HSA_STATUS_SUCCESS ||
					    segment != HSA_AMD_SEGMENT_GLOBAL)
						return llvm::Error::success();
					check(amd.callFunction<
						hsa_amd_memory_pool_get_info>(
						candidate,
						HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
						&flags));
					check(amd.callFunction<
						hsa_amd_memory_pool_get_info>(
						candidate,
						HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
						&alloc));
					check(amd.callFunction<
						hsa_amd_agent_memory_pool_get_info>(
						agent, candidate,
						HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
						&access));
					if (!(flags &
					      (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
					       HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) ||
					    !alloc ||
					    access ==
						    HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
						return llvm::Error::success();
					check(amd.callFunction<
						hsa_amd_agent_memory_pool_get_info>(
						agent, candidate,
						HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS,
						&hops));
					std::vector<
						hsa_amd_memory_pool_link_info_t>
						links(hops);
					if (hops)
						check(amd.callFunction<
							hsa_amd_agent_memory_pool_get_info>(
							agent, candidate,
							HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO,
							links.data()));
					for (const auto &link : links)
						if (!link.atomic_support_32bit ||
						    !link.atomic_support_64bit ||
						    !link.coherent_support)
							return llvm::Error::
								success();
					pool = candidate;
					return llvm::Error::success();
				} catch (const std::exception &error) {
					return llvm::make_error<
						llvm::StringError>(
						error.what(),
						llvm::inconvertibleErrorCode());
				} catch (...) {
					return llvm::make_error<
						llvm::StringError>(
						"Cannot inspect ROCm memory pool",
						llvm::inconvertibleErrorCode());
				}
			});
		if (error)
			throw std::runtime_error(
				llvm::toString(std::move(error)));
		if (pool.handle)
			break;
	}
	if (!pool.handle)
		throw std::runtime_error(
			"No CPU/GPU coherent pool with system atomics for ROCm maps");
	try {
		uint64_t frequency = 0;
		check(core.callFunction<hsa_agent_get_info>(
			agent,
			static_cast<hsa_agent_info_t>(
				HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY),
			&frequency));
		if (!frequency || frequency > UINT64_MAX / 5)
			throw std::runtime_error(
				"Invalid GPU timestamp frequency");
		void *allocation = nullptr;
		check(amd.callFunction<hsa_amd_memory_pool_allocate>(
			pool, sizeof(*data), 0, &allocation));
		data = new (allocation) rocm_channel_data{};
		data->timeout_ticks = frequency * 5;
		check(amd.callFunction<hsa_amd_agents_allow_access>(
			1, &agent, nullptr, data));
		check(amd.callFunction<hsa_amd_memory_lock_to_pool>(
			first.segment_base, first.segment_size, &agent, 1, pool,
			0, &device_base));
		host_base = first.segment_base;
		segment_size = first.segment_size;
		worker = std::thread([this] { serve(); });
	} catch (...) {
		if (host_base)
			amd.callFunction<hsa_amd_memory_unlock>(host_base);
		if (data)
			amd.callFunction<hsa_amd_memory_pool_free>(data);
		throw;
	}
}
rocm_helper_channel::~rocm_helper_channel()
{
	__atomic_store_n(&data->stopped, 1, __ATOMIC_RELEASE);
	if (worker.joinable())
		worker.join();
	amd.callFunction<hsa_amd_memory_unlock>(host_base);
	amd.callFunction<hsa_amd_memory_pool_free>(data);
}
uint64_t rocm_helper_channel::bind(const rocm_map_binding &map)
{
	const unsigned count =
		__atomic_load_n(&data->map_count, __ATOMIC_ACQUIRE);
	for (unsigned i = 0; i < count; ++i)
		if (maps[i].fd == map.fd &&
		    maps[i].info.extra_buffer == map.info.extra_buffer)
			return i + 1;
	if (count == rocm_max_maps || map.segment_base != host_base ||
	    map.segment_size != segment_size || !map.invoke ||
	    map.info.value_size <= 0 ||
	    map.info.value_size > int(rocm_max_value_size))
		throw std::runtime_error(
			"Unsupported ROCm map binding or channel capacity");
	maps[count] = map;
	data->value_sizes[count] = map.info.value_size;
	__atomic_store_n(&data->map_count, count + 1, __ATOMIC_RELEASE);
	return count + 1;
}
void rocm_helper_channel::serve() noexcept
{
	while (active()) {
		if (__atomic_load_n(&data->state, __ATOMIC_ACQUIRE) !=
		    rocm_request) {
			std::this_thread::sleep_for(
				std::chrono::microseconds(20));
			continue;
		}
		const auto op = data->operation;
		uint64_t result = op == 1 ? 0 : uint64_t(-EINVAL);
		try {
			if (data->map < __atomic_load_n(&data->map_count,
							__ATOMIC_ACQUIRE) &&
			    op >= 1 && op <= 3) {
				const auto &binding = maps[data->map];
				result = binding.invoke(op, data->key,
							data->value,
							data->flags);
				if (op == 1 && result) {
					const auto base =
						reinterpret_cast<uint64_t>(
							host_base);
					const uint64_t size =
						binding.info.value_size;
					if (result < base ||
					    size > segment_size ||
					    result - base > segment_size - size)
						result = 0;
					else
						result = reinterpret_cast<
								 uint64_t>(
								 device_base) +
							 result - base;
				}
			}
		} catch (...) {
			result = op == 1 ? 0 : uint64_t(-EIO);
		}
		data->result = result;
		__atomic_store_n(&data->state, rocm_response, __ATOMIC_RELEASE);
	}
}
} // namespace bpftime::attach
