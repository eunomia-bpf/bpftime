/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/map_common_def.hpp"
#include "linux/bpf.h"
#include "spdlog/fmt/bin_to_hex.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include "platform_utils.hpp"


namespace bpftime
{
per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	: per_cpu_hash_map_impl(memory, key_size, value_size, max_entries,
				sysconf(_SC_NPROCESSORS_ONLN))
{
}

per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries, int ncpu)
	: impl(memory.get_segment_manager()), key_size(key_size),
	  value_size(value_size), ncpu(std::max(1, ncpu)),
	  _max_entries(max_entries),
	  key_template(key_size, memory.get_segment_manager()),
	  value_template(size_t(value_size) * this->ncpu,
			 memory.get_segment_manager())
{
	SPDLOG_DEBUG(
		"Initializing per cpu hash, key size {}, value size {}, ncpu {}",
		key_size, value_size, this->ncpu);
}

void *per_cpu_hash_map_impl::elem_lookup(const void *key)
{
	int cpu = my_sched_getcpu();
	SPDLOG_DEBUG("Run per cpu hash lookup at cpu {}", cpu);
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	if (cpu < 0 || cpu >= ncpu) {
		errno = EINVAL;
		return nullptr;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	if (auto itr = impl.find(key_template); itr != impl.end()) {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		return &itr->second[size_t(value_size) * cpu];
	} else {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long per_cpu_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	if (key == nullptr || value == nullptr) {
		errno = EINVAL;
		return -1;
	}
	flags &= 0xFFFFFFFFULL;
	int cpu = my_sched_getcpu();
	if (cpu < 0 || cpu >= ncpu) {
		errno = EINVAL;
		return -1;
	}
	SPDLOG_DEBUG(
		"Per cpu update, key {}, value {}",
		spdlog::to_hex((const uint8_t *)key,
			       (const uint8_t *)key + key_size),
		spdlog::to_hex((const uint8_t *)value,
			       (const uint8_t *)value + value_size));

	SPDLOG_DEBUG("Run per cpu hash update at cpu {}", cpu);
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	auto itr = impl.find(key_template);
	bool elem_exists = itr != impl.end();
	// Enforce BPF update-flag and capacity semantics, matching the
	// userspace and var-size hash implementations.
	if (flags == BPF_NOEXIST && elem_exists) {
		errno = EEXIST;
		return -1;
	}
	if (flags == BPF_EXIST && !elem_exists) {
		errno = ENOENT;
		return -1;
	}
	if (!elem_exists && impl.size() == _max_entries) {
		errno = E2BIG;
		return -1;
	}
	const auto *val_bytes = static_cast<const uint8_t *>(value);
	if (elem_exists) {
		std::copy(val_bytes, val_bytes + value_size,
			  itr->second.begin() + size_t(cpu) * value_size);
	} else {
		bytes_vec full_value_vec = this->value_template;
		std::copy(val_bytes, val_bytes + value_size,
			  full_value_vec.begin() + size_t(cpu) * value_size);
		impl.insert(bi_map_value_ty(key_template, full_value_vec));
	}

	return 0;
}

long per_cpu_hash_map_impl::elem_delete(const void *key)
{
	if (key == nullptr) {
		errno = ENOENT;
		return -1;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	auto itr = impl.find(key_template);
	if (itr == impl.end()) {
		errno = ENOENT;
		return -1;
	}
	// bpf_map_delete_elem removes the whole element regardless of the
	// current cpu slot; the old per-cpu fill targeted the wrong range and
	// left the key lookupable.
	impl.erase(itr);
	return 0;
}

int per_cpu_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr) {
		// nullptr means the first key
		auto itr = impl.begin();
		if (itr == impl.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);

	auto itr = impl.find(key_template);
	if (itr == impl.end()) {
		// not found, should be refer to the first key
		return map_get_next_key(nullptr, next_key);
	}
	itr++;
	if (itr == impl.end()) {
		// If *key* is the last element, returns -1 and *errno*
		// is set to **ENOENT**.
		errno = ENOENT;
		return -1;
	}
	std::copy(itr->first.begin(), itr->first.end(), (uint8_t *)next_key);
	return 0;
}
void *per_cpu_hash_map_impl::elem_lookup_userspace(const void *key)
{
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	if (auto itr = impl.find(key_template); itr != impl.end()) {
		SPDLOG_TRACE("Exit elem lookup of hash map: {}",
			     spdlog::to_hex(itr->second.begin(),
					    itr->second.end()));
		return &itr->second[0];
	} else {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long per_cpu_hash_map_impl::elem_update_userspace(const void *key,
						  const void *value,
						  uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	if (key == nullptr || value == nullptr) {
		errno = EINVAL;
		return -1;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	bytes_vec value_vec = this->value_template;
	value_vec.assign((const uint8_t *)value,
			 (const uint8_t *)value + size_t(value_size) * ncpu);
	bool elem_exists = impl.find(key_template) != impl.end();
	if (flags == BPF_NOEXIST && elem_exists) {
		errno = EEXIST;
		return -1;
	}
	if (flags == BPF_EXIST && !elem_exists) {
		errno = ENOENT;
		return -1;
	}
	if (elem_exists == false && impl.size() == _max_entries) {
		errno = E2BIG;
		return -1;
	}
	impl.insert_or_assign(key_template, value_vec);
	return 0;
}
long per_cpu_hash_map_impl::elem_delete_userspace(const void *key)
{
	return elem_delete(key);
}

long per_cpu_hash_map_impl::lookup_and_delete_userspace(const void *key,
							void *value)
{
	if (key == nullptr || value == nullptr) {
		errno = EINVAL;
		return -1;
	}
	key_template.assign((const uint8_t *)key,
			    (const uint8_t *)key + key_size);
	auto itr = this->impl.find(key_template);
	if (itr == impl.end()) {
		errno = ENOENT;
		return -1;
	}
	memcpy(value, itr->second.data(), size_t(ncpu) * value_size);
	impl.erase(itr);
	return 0;
}
} // namespace bpftime
