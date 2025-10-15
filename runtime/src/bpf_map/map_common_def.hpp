/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_COMMON_DEF_HPP
#define _MAP_COMMON_DEF_HPP
#include "spdlog/spdlog.h"
#include <boost/container_hash/hash.hpp>
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <cstdint>
#include <functional>
#include "platform_utils.hpp"

namespace bpftime
{
// Extended update-op flags for bpftime GPU (stored in high 32 bits of flags)
// This avoids collision with kernel BPF_ANY/BPF_NOEXIST/BPF_EXIST (low 32 bits)
static constexpr uint64_t BPFTIME_UPDATE_OP_SHIFT = 32ULL;
static constexpr uint64_t BPFTIME_UPDATE_OP_MASK = 0xFFFFFFFFULL
						   << BPFTIME_UPDATE_OP_SHIFT;
static constexpr uint64_t BPFTIME_UPDATE_OP_NONE = 0ULL;
static constexpr uint64_t BPFTIME_UPDATE_OP_ADD = 1ULL
						  << BPFTIME_UPDATE_OP_SHIFT;
// Usage of extended-update flags:
// - The low 32 bits keep the standard kernel flags (e.g., BPF_ANY/BPF_NOEXIST/BPF_EXIST).
// - The high 32 bits encode a single bpftime-specific operation such as
//   BPFTIME_UPDATE_OP_ADD to request a host-side fetch_add.
// Semantics of BPFTIME_UPDATE_OP_ADD (fetch_add):
// - The 'value' pointer provided to map_update_elem is treated as an unsigned
//   64-bit delta.
// - The map backend performs a software read-modify-write on the targeted
//   element (load current u64, add delta, store back).
// Concurrency:
// - This is NOT a hardware-atomic RMW across threads/processes; without
//   additional serialization, concurrent writers could lose updates.
// - Deploy a higher-level gate (e.g., leader-thread guard) when multiple
//   writers may update the same key.

using bytes_vec_allocator = boost::interprocess::allocator<
	uint8_t, boost::interprocess::managed_shared_memory::segment_manager>;
using bytes_vec = boost::interprocess::vector<uint8_t, bytes_vec_allocator>;
using uint64_vec_allocator = boost::interprocess::allocator<
	uint64_t, boost::interprocess::managed_shared_memory::segment_manager>;
using uint64_vec = boost::interprocess::vector<uint64_t, uint64_vec_allocator>;

template <class T>
static inline T ensure_on_current_cpu(std::function<T(int cpu)> func)
{
	return func(my_sched_getcpu());
}

template <class T>
static inline T ensure_on_certain_cpu(int cpu, std::function<T()> func)
{
	static thread_local int currcpu = -1;
	if (currcpu == my_sched_getcpu()) {
		return func(currcpu);
	}
	cpu_set_t orig, set;
	CPU_ZERO(&orig);
	CPU_ZERO(&set);
	sched_getaffinity(0, sizeof(orig), &orig);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
	T ret = func();
	sched_setaffinity(0, sizeof(orig), &orig);
	return ret;
}

template <>
inline void ensure_on_certain_cpu(int cpu, std::function<void()> func)
{
	cpu_set_t orig, set;
	CPU_ZERO(&orig);
	CPU_ZERO(&set);
	sched_getaffinity(0, sizeof(orig), &orig);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
	func();
	sched_setaffinity(0, sizeof(orig), &orig);
}

struct bytes_vec_hasher {
	size_t operator()(bytes_vec const &vec) const
	{
		using boost::hash_combine;
		size_t seed = 0;
		hash_combine(seed, vec.size());
		for (auto x : vec)
			hash_combine(seed, x);
		return seed;
	}
};

struct uint32_hasher {
	size_t operator()(uint32_t const &data) const
	{
		return data;
	}
};

static inline bool check_update_flags(uint64_t flags)
{
	// Allow custom bpftime ops in the high 32 bits; validate only low 32
	// bits
	uint64_t base_flags = flags & 0xFFFFFFFFULL;
	if (base_flags != 0 /*BPF_ANY*/ && base_flags != 1 /*BPF_NOEXIST*/ &&
	    base_flags != 2 /*BPF_EXIST*/) {
		errno = EINVAL;
		return false;
	}
	return true;
}
struct int_hasher {
	size_t operator()(int const &data) const
	{
		return data;
	}
};
} // namespace bpftime

#endif
