/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/map_common_def.hpp"
#include "linux/bpf.h"
#include "platform_utils.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include <cerrno>
#include <unistd.h>

namespace bpftime
{
per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	: per_cpu_hash_map_impl(memory, key_size, value_size, max_entries,
				bpftime_percpu_columns())
{
}

per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries, int ncpu)
	: impl(memory, max_entries, key_size,
	       size_t((value_size + 7U) & ~7U) * std::max(1, ncpu)),
	  value_size(value_size), ncpu(std::max(1, ncpu))
{
	SPDLOG_DEBUG(
		"Initializing per cpu hash, key size {}, value size {}, ncpu {}",
		key_size, value_size, this->ncpu);
}

void *per_cpu_hash_map_impl::elem_lookup(const void *key)
{
	int cpu = my_sched_getcpu();
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	if (cpu < 0 || cpu >= ncpu) {
		errno = EINVAL;
		return nullptr;
	}
	void *base = impl.elem_lookup(key);
	if (base == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	return static_cast<uint8_t *>(base) + size_t(stride()) * cpu;
}

long per_cpu_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	flags &= 0xFFFFFFFFULL;
	int cpu = my_sched_getcpu();
	if (cpu < 0 || cpu >= ncpu) {
		errno = EINVAL;
		return -1;
	}
	return impl.upsert(key, value, flags, false, size_t(cpu) * stride(),
			   value_size);
}

long per_cpu_hash_map_impl::elem_delete(const void *key)
{
	return impl.erase(key, nullptr, false);
}

int per_cpu_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	return impl.map_get_next_key(key, next_key);
}

void *per_cpu_hash_map_impl::elem_lookup_userspace(const void *key)
{
	void *base = impl.elem_lookup(key);
	if (base == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	return base;
}

long per_cpu_hash_map_impl::elem_update_userspace(const void *key,
						  const void *value,
						  uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	flags &= 0xFFFFFFFFULL;
	return impl.upsert(key, value, flags, true, 0,
			   size_t(stride()) * ncpu);
}

long per_cpu_hash_map_impl::elem_delete_userspace(const void *key)
{
	return impl.erase(key, nullptr, true);
}

long per_cpu_hash_map_impl::lookup_and_delete_userspace(const void *key,
							void *value,
							bool from_syscall)
{
	if (value == nullptr) {
		errno = EINVAL;
		return -1;
	}
	return impl.erase(key, value, from_syscall);
}
} // namespace bpftime
