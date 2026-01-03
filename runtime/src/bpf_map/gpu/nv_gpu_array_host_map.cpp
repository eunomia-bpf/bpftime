#include "nv_gpu_array_host_map.hpp"
#include "bpftime_internal.h"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda.h"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

using namespace bpftime;

CUdeviceptr
nv_gpu_array_host_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (shm_holder.global_shared_memory.get_open_type() ==
	    shm_open_type::SHM_OPEN_ONLY) {
		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr == agent_gpu_shared_mem.end()) {
			SPDLOG_INFO(
				"Initializing nv_gpu_array_host_map_impl at pid {}, "
				"shared mem addr={:x}",
				pid, (uintptr_t)data_buffer.data());

			void *cpu_ptr = data_buffer.data();
			CUdeviceptr gpu_ptr;
			auto err =
				cuMemHostGetDevicePointer(&gpu_ptr, cpu_ptr, 0);
			if (err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to convert cpu ptr to gpu ptr: {}",
					(int)err);
			}
			if (gpu_ptr == 0) {
				SPDLOG_ERROR(
					"Failed to convert CPU pointer to GPU pointer for per-GPU-thread array host map");
				throw std::runtime_error(
					"Failed to convert CPU pointer to GPU pointer!");
			}
			SPDLOG_INFO(
				"Mapped shared memory for GPU access: cpu={:x}, gpu={:x}",
				(uintptr_t)data_buffer.data(),
				(uintptr_t)gpu_ptr);
			agent_gpu_shared_mem[pid] = (CUdeviceptr)gpu_ptr;
		}
		return agent_gpu_shared_mem[pid];
	} else {
		return (CUdeviceptr)data_buffer.data();
	}
}

nv_gpu_array_host_map_impl::nv_gpu_array_host_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint64_t value_size,
	uint64_t max_entries, uint64_t thread_count)
	: data_buffer(memory.get_segment_manager()),
	  agent_gpu_shared_mem(memory.get_segment_manager()),
	  value_size(value_size), max_entries(max_entries),
	  thread_count(thread_count > 0 ? thread_count : 1) // Default to 1 if
							    // not specified
{
	entry_size = this->thread_count * value_size;
	auto total_buffer_size =
		(uint64_t)value_size * max_entries * this->thread_count;
	SPDLOG_INFO(
		"Initializing map type of BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP (shared mem), "
		"value_size={}, max_entries={}, thread_count={}, total_buffer_size={}",
		value_size, max_entries, thread_count, total_buffer_size);

	// Allocate in boost::interprocess shared memory
	data_buffer.resize(total_buffer_size);
	std::memset(data_buffer.data(), 0, total_buffer_size);

	SPDLOG_INFO("Allocated shared memory buffer at {:x}",
		    (uintptr_t)data_buffer.data());
}

void *nv_gpu_array_host_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(uint32_t *)key;
	if (key_val >= max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	// Memory barrier: ensure we see latest GPU writes
	std::atomic_thread_fence(std::memory_order_acquire);
	return data_buffer.data() + (uint64_t)key_val * entry_size;
}

long nv_gpu_array_host_map_impl::elem_update(const void *key, const void *value,
					     uint64_t flags)
{
	if (unlikely(!check_update_flags(flags)))
		return -1;
	auto key_val = *(uint32_t *)key;
	if (unlikely(key_val < max_entries && flags == BPF_NOEXIST)) {
		errno = EEXIST;
		return -1;
	}

	if (unlikely(key_val >= max_entries)) {
		errno = E2BIG;
		return -1;
	}
	std::memcpy(data_buffer.data() + (uint64_t)key_val * entry_size, value,
		    entry_size);
	// Memory barrier: ensure CPU writes are visible to GPU
	std::atomic_thread_fence(std::memory_order_release);
	return 0;
}

long nv_gpu_array_host_map_impl::elem_delete(const void *key)
{
	errno = EINVAL;
	return -1;
}

int nv_gpu_array_host_map_impl::map_get_next_key(const void *key,
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

nv_gpu_array_host_map_impl::~nv_gpu_array_host_map_impl()
{
	SPDLOG_DEBUG("Destroying nv_gpu_array_host_map_impl");
}
