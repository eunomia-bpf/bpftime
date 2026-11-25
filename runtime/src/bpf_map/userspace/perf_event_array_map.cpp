/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <bpf_map/userspace/perf_event_array_map.hpp>
#include <cerrno>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace bpftime
{
perf_event_array_map_impl::perf_event_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	// Default value of fd map is -1
	: data(max_entries, -1, memory.get_segment_manager())
{
	if (key_size != 4 || value_size != 4) {
		SPDLOG_ERROR(
			"Key size and value size of perf_event_array must be 4");
		throw std::runtime_error(
			"Key size and value size of perf_event_array must be 4");
	}
}

void *perf_event_array_map_impl::elem_lookup(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return nullptr;
	}
	return &data[k];
}

long perf_event_array_map_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	int32_t v = *(int32_t *)value;
	data[k] = v;
	return 0;
}

long perf_event_array_map_impl::elem_delete(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	data[k] = -1;
	return 0;
}

int perf_event_array_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Kernel array_map_get_next_key semantics: an absent or out-of-range
	// key restarts at index 0 instead of failing, and the last index is
	// the only one that reports -ENOENT. Array iteration is dense, so
	// slots cleared by elem_delete are still visited.
	uint32_t index = key != nullptr ? *(const uint32_t *)key : UINT32_MAX;
	uint32_t *out = (uint32_t *)next_key;

	if (index >= data.size()) {
		*out = 0;
		return 0;
	}
	if (index == data.size() - 1) {
		errno = ENOENT;
		return -1;
	}
	*out = index + 1;
	return 0;
}

} // namespace bpftime
