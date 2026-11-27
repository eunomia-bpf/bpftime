// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include "rocm_channel_abi.hpp"
using namespace bpftime::attach;

// Resolved to the owning channel's device address by the IR linker.
extern "C" __device__ unsigned long long bpftime_rocm_channel_address();

static __device__ uint32_t load_control(uint32_t *ptr)
{
	return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE,
				 __HIP_MEMORY_SCOPE_SYSTEM);
}
static __device__ void store_control(uint32_t *ptr, uint32_t value)
{
	__hip_atomic_store(ptr, value, __ATOMIC_RELEASE,
			   __HIP_MEMORY_SCOPE_SYSTEM);
}
static __device__ uint64_t map_call(unsigned op, uint64_t map, uint64_t key,
				    uint64_t value, uint64_t flags)
{
	auto *channel = reinterpret_cast<rocm_channel_data *>(
		bpftime_rocm_channel_address());
	const uint64_t failure = op == 1 ? 0 : uint64_t(-110); // ETIMEDOUT
	if (!channel || load_control(&channel->stopped))
		return failure;
	if (!map || map > load_control(&channel->map_count) || !key ||
	    (op == 2 && !value))
		return op == 1 ? 0 : uint64_t(-22);
	const unsigned slot = map - 1;
	const uint64_t start = wall_clock64();
	auto expired = [&]() {
		if (load_control(&channel->stopped))
			return true;
		if (uint64_t(wall_clock64()) - start < channel->timeout_ticks)
			return false;
		// Poison the entire channel. A late CPU response must never
		// race with a new request reusing this slot.
		store_control(&channel->stopped, 1);
		return true;
	};
	for (;;) {
		if (expired())
			return failure;
		uint32_t expected = 0;
		if (__hip_atomic_compare_exchange_strong(
			    &channel->lock, &expected, 1, __ATOMIC_ACQUIRE,
			    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM))
			break;
		__builtin_amdgcn_s_sleep(1);
	}
	channel->operation = op;
	channel->map = slot;
	channel->key = *reinterpret_cast<const uint32_t *>(key);
	channel->flags = flags;
	if (op == 2)
		for (unsigned i = 0; i < channel->value_sizes[slot]; ++i)
			channel->value[i] =
				reinterpret_cast<const uint8_t *>(value)[i];
	store_control(&channel->state, rocm_request);
	while (load_control(&channel->state) != rocm_response) {
		if (expired())
			return failure;
		__builtin_amdgcn_s_sleep(1);
	}
	const auto result = channel->result;
	store_control(&channel->state, rocm_idle);
	store_control(&channel->lock, 0);
	return result;
}

extern "C" __device__ __attribute__((used)) unsigned long long
_bpf_helper_ext_0001(unsigned long long map, unsigned long long key,
		     unsigned long long, unsigned long long, unsigned long long)
{
	return map_call(1, map, key, 0, 0);
}
extern "C" __device__ __attribute__((used)) unsigned long long
_bpf_helper_ext_0002(unsigned long long map, unsigned long long key,
		     unsigned long long value, unsigned long long flags,
		     unsigned long long)
{
	return map_call(2, map, key, value, flags);
}
extern "C" __device__ __attribute__((used)) unsigned long long
_bpf_helper_ext_0003(unsigned long long map, unsigned long long key,
		     unsigned long long, unsigned long long, unsigned long long)
{
	return map_call(3, map, key, 0, 0);
}
// Retain this definition during device-only compilation. The IR linker
// externalizes it before resolving the eBPF helper declaration.
extern "C" __device__ __attribute__((used)) unsigned long long
_bpf_helper_ext_0506(unsigned long long str, unsigned long long,
		     unsigned long long, unsigned long long, unsigned long long)
{
	return printf("%s\n", reinterpret_cast<const char *>(str));
}
