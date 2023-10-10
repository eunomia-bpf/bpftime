#include "spdlog/spdlog.h"
#include <bpf_map/perf_event_array_map.hpp>
#include <cassert>
#include <cerrno>
#include <spdlog/spdlog.h>
namespace bpftime
{
perf_event_array_map_impl::perf_event_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	: data(max_entries, memory.get_segment_manager())
{
	if (key_size != 4 || value_size != 4) {
		spdlog::error(
			"Key size and value size of perf_event_array must be 4");
		assert(false);
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
	spdlog::error(
		"Try to call elem_delete of perf_event_array, which is not supported");
	errno = ENOTSUP;
	return -1;
}

int perf_event_array_map_impl::bpf_map_get_next_key(const void *key, void *next_key)
{
	int32_t *out = (int32_t *)next_key;
	if (key == nullptr) {
		*out = 0;
		return 0;
	}
	int32_t k = *(int32_t *)key;
	// The last key
	if ((size_t)(k + 1) == data.size()) {
		errno = ENOENT;
		return -1;
	}
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	*out = k + 1;
	return 0;
}

} // namespace bpftime
