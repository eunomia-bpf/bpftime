/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/map_common_def.hpp"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/userspace/per_cpu_array_map.hpp>
#include <cerrno>
#include <unistd.h>

namespace bpftime
{

per_cpu_array_map_impl::per_cpu_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t value_size,
	uint32_t max_entries, uint32_t ncpu)
	: data(memory.get_segment_manager()), ncpu(ncpu),
	  value_size(value_size), max_ent(max_entries)
{
	data.resize(max_entries * ncpu * value_size);
}

per_cpu_array_map_impl::per_cpu_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t value_size,
	uint32_t max_entries)
	: per_cpu_array_map_impl(memory, value_size, max_entries,
				 sysconf(_SC_NPROCESSORS_ONLN))
{
}

void *per_cpu_array_map_impl::elem_lookup(const void *key)
{
	return ensure_on_current_cpu<void *>([&](int cpu) -> void * {
		if (key == nullptr) {
			errno = ENOENT;
			return nullptr;
		}
		uint32_t key_val = *(uint32_t *)key;
		if (key_val >= max_ent) {
			errno = ENOENT;
			return nullptr;
		}
		return data_at(key_val, cpu);
	});
}

long per_cpu_array_map_impl::elem_update(const void *key, const void *value,
					 uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	return ensure_on_current_cpu<long>([&](int cpu) -> long {
		// return impl[cpu].elem_update(key, value, flags);
		if (key == nullptr) {
			errno = ENOENT;
			return -1;
		}
		uint32_t key_val = *(uint32_t *)key;
		if (key_val < max_ent && flags == BPF_NOEXIST) {
			errno = EEXIST;
			return -1;
		}
		if (key_val >= max_ent) {
			errno = E2BIG;
			return -1;
		}
		std::copy((uint8_t *)value, (uint8_t *)value + value_size,
			  data_at(key_val, cpu));
		return 0;
	});
}

long per_cpu_array_map_impl::elem_delete(const void *key)
{
	errno = EINVAL;
	SPDLOG_DEBUG("Deleting of per cpu array is not supported");
	return -1;
}

int per_cpu_array_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Not found
	if (key == nullptr || *(uint32_t *)key >= max_ent) {
		*(uint32_t *)next_key = 0;
		return 0;
	}
	uint32_t deref_key = *(uint32_t *)key;
	// Last element
	if (deref_key == max_ent - 1) {
		errno = ENOENT;
		return -1;
	}
	auto key_val = *(uint32_t *)key;
	*(uint32_t *)next_key = key_val + 1;
	return 0;
}

void *per_cpu_array_map_impl::elem_lookup_userspace(const void *key)
{
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	uint32_t key_val = *(uint32_t *)key;
	if (key_val >= max_ent) {
		errno = ENOENT;
		return nullptr;
	}
	return data_at(key_val, 0);
}

long per_cpu_array_map_impl::elem_update_userspace(const void *key,
						   const void *value,
						   uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	if (key == nullptr) {
		errno = ENOENT;
		return -1;
	}
	uint32_t key_val = *(uint32_t *)key;
	if (key_val < max_ent && flags == BPF_NOEXIST) {
		errno = EEXIST;
		return -1;
	}
	if (key_val >= max_ent) {
		errno = E2BIG;
		return -1;
	}
	std::copy((uint8_t *)value, (uint8_t *)value + ncpu * value_size,
		  data_at(key_val, 0));
	return 0;
}

long per_cpu_array_map_impl::elem_delete_userspace(const void *key)
{
	errno = EINVAL;
	SPDLOG_WARN("Element delete is not supported by per cpu array");
	return -1;
}

} // namespace bpftime
