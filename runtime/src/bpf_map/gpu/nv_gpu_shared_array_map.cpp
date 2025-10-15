#include "nv_gpu_shared_array_map.hpp"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include "bpf_map/map_common_def.hpp"
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <unistd.h>

using namespace bpftime;

nv_gpu_shared_array_map_impl::nv_gpu_shared_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint64_t value_size,
	uint64_t max_entries)
	: value_size(value_size), max_entries(max_entries),
	  shared_area(memory.get_segment_manager()),
	  value_buffer(memory.get_segment_manager())
{
	value_buffer.resize(value_size);
	// Allocate backing bytes in bpftime shared memory segment
	uint64_t total = align_up(max_entries * sizeof(int) /*locks*/ +
					  max_entries * sizeof(int) /*dirty*/,
				  8) +
			 value_size * max_entries;
	shared_area.resize(total);
	std::fill(shared_area.begin(), shared_area.end(), 0);
	SPDLOG_INFO(
		"Initializing map type of BPF_MAP_TYPE_GPU_ARRAY_MAP (host-shared), total_buffer_size={}",
		total);
}

void *nv_gpu_shared_array_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(uint32_t *)key;
	if (key_val >= max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	// Directly read from shared memory (already registered to CUDA on
	// agent)
	auto src = values_region_base() + key_val * value_size;
	std::memcpy(value_buffer.data(), src, value_size);
	return value_buffer.data();
}

long nv_gpu_shared_array_map_impl::elem_update(const void *key,
					       const void *value,
					       uint64_t flags)
{
	if (!bpftime::check_update_flags(flags))
		return -1;
	auto key_val = *(uint32_t *)key;
	if ((key_val < max_entries) && flags == BPF_NOEXIST) {
		errno = EEXIST;
		return -1;
	}
	if (key_val >= max_entries) {
		errno = E2BIG;
		return -1;
	}
    // Host writes to shared memory; GPU sees via UVA (zero-copy). For
    // non-per-thread GPU array map, host-side update is a plain overwrite.
    auto dst = values_region_base() + key_val * value_size;
    std::memcpy(dst, value, value_size);
    return 0;
}

long nv_gpu_shared_array_map_impl::elem_delete(const void *key)
{
	errno = EINVAL;
	return -1;
}

int nv_gpu_shared_array_map_impl::map_get_next_key(const void *key,
						   void *next_key)
{
	auto &next_key_val = *(uint32_t *)next_key;

	if (key == nullptr) {
		next_key_val = 0;
		return 0;
	} else {
		auto key_val = *(uint32_t *)key;
		if (key_val >= max_entries) {
			next_key = 0;
			return 0;
		}
		if (key_val + 1 == max_entries) {
			errno = ENOENT;
			return -1;
		}
		next_key_val = key_val + 1;
		return 0;
	}
}

nv_gpu_shared_array_map_impl::~nv_gpu_shared_array_map_impl()
{
	// shared_area is managed by boost::interprocess and will be released
	// with segment
}
