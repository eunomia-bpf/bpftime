/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/map_common_def.hpp"
#include "spdlog/fmt/bin_to_hex.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include <unistd.h>

namespace bpftime
{
per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size)
	: per_cpu_hash_map_impl(memory, key_size, value_size,
				sysconf(_SC_NPROCESSORS_ONLN))
{
}

per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, int ncpu)
	: impl_vec{ memory.get_segment_manager() }, key_size(key_size),
	  value_size(value_size), ncpu(ncpu),
	  key_buffers{ memory.get_segment_manager() },
	  value_buffers{ memory.get_segment_manager() },
	  lookup_value_res{ ncpu * value_size, memory.get_segment_manager() }
{
	SPDLOG_DEBUG(
		"Initializing per cpu hash, key size {}, value size {}, ncpu {}",
		key_size, value_size, ncpu);
	for (int i = 0; i < ncpu; i++) {
		impl_vec.push_back(shm_hash_map(memory.get_segment_manager()));
		key_buffers.push_back(
			bytes_vec(key_size, memory.get_segment_manager()));
		value_buffers.push_back(
			bytes_vec(value_size, memory.get_segment_manager()));
	}
}

void *per_cpu_hash_map_impl::elem_lookup(const void *key)
{
	int cpu = sched_getcpu();
	SPDLOG_DEBUG("Run per cpu hash lookup at cpu {}", cpu);
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	auto cur_map = impl_vec[cpu];
	auto &key_vec = key_buffers[cpu];
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	if (auto itr = cur_map.find(key_vec); itr != cur_map.end()) {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		return itr->second.data();
	} else {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long per_cpu_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	SPDLOG_DEBUG("Per cpu update, key {}, value {}", (const char *)key,
		     *(long *)value);
	int cpu = sched_getcpu();
	SPDLOG_DEBUG("Run per cpu hash update at cpu {}", cpu);
	auto cur_map = impl_vec[cpu];
	auto &key_vec = key_buffers[cpu];
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	auto &value_vec = value_buffers[cpu];
	value_vec.assign((uint8_t *)value,
			 (uint8_t *)value + value_size);
	if (auto itr = cur_map.find(key_vec); itr != cur_map.end()) {
		itr->second = value_vec;
	} else {
		cur_map.insert(bi_map_value_ty(key_vec, value_vec));
	}
	return 0;
}

long per_cpu_hash_map_impl::elem_delete(const void *key)
{
	int cpu = sched_getcpu();
	SPDLOG_DEBUG("Run per cpu hash delete at cpu {}", cpu);
	auto cur_map = impl_vec[cpu];
	auto &key_vec = key_buffers[cpu];
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	cur_map.erase(key_vec);
	return 0;
}

int per_cpu_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr) {
		get_next_key_cur_cpu = 0;
		// nullptr means the first key
		auto cur_map = impl_vec[0];
		auto itr = cur_map.begin();
		if (itr == cur_map.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	auto &key_vec = key_buffers[sched_getcpu()];
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	bool found = false;
	// we use global variable to store the current cpu for access maps
	for (; get_next_key_cur_cpu < ncpu; get_next_key_cur_cpu++) {
		auto cur_map = impl_vec[get_next_key_cur_cpu];
		auto itr = cur_map.find(key_vec);
		if (itr != cur_map.end()) {
			found = true;
			itr++;
			if (itr == cur_map.end() &&
			    get_next_key_cur_cpu == ncpu - 1) {
				// If *key* is the last element and cpu is the
				// last cpu, returns -1 and *errno* is set to
				// **ENOENT**.
				errno = ENOENT;
				return -1;
			}
			std::copy(itr->first.begin(), itr->first.end(),
				  (uint8_t *)next_key);
			return 0;
		} else {
			// not found, let's try next cpu
		}
	}
	if (!found) {
		// not found, should be refer to the first key
		get_next_key_cur_cpu = 0;
		return map_get_next_key(nullptr, next_key);
	}
	return 0;
}

void *per_cpu_hash_map_impl::elem_lookup_userspace(const void *key)
{
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	lookup_value_res.assign(value_size * ncpu, 0);
	for (int cpu = 0; cpu < ncpu; cpu++) {
		auto cur_map = impl_vec[cpu];
		bytes_vec key_vec = key_buffers[sched_getcpu()];
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		if (auto itr = cur_map.find(key_vec); itr != cur_map.end()) {
			std::copy(itr->second.begin(), itr->second.end(),
				  lookup_value_res.begin() + cpu * value_size);
		}
	}
	return lookup_value_res.data();
}

long per_cpu_hash_map_impl::elem_update_userspace(const void *key,
						  const void *value,
						  uint64_t flags)
{
	for (int cpu = 0; cpu < ncpu; cpu++) {
		auto cur_map = impl_vec[cpu];
		// make a copy of value to avoid locks, since the vec may be
		// shared
		bytes_vec key_vec = key_buffers[sched_getcpu()];
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		bytes_vec value_vec = value_buffers[sched_getcpu()];
		void *cpu_value = (void *)((uint8_t *)value + cpu * value_size);
		value_vec.assign((uint8_t *)cpu_value,
				 (uint8_t *)cpu_value + value_size);
		if (auto itr = cur_map.find(key_vec); itr != cur_map.end()) {
			itr->second = value_vec;
		} else {
			cur_map.insert(bi_map_value_ty(key_vec, value_vec));
		}
	}
	return 0;
}

long per_cpu_hash_map_impl::elem_delete_userspace(const void *key)
{
	for (int cpu = 0; cpu < ncpu; cpu++) {
		auto cur_map = impl_vec[cpu];
		auto &key_vec = key_buffers[sched_getcpu()];
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		cur_map.erase(key_vec);
	}
	return 0;
}
} // namespace bpftime
