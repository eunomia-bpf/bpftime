/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "array_map_kernel_gpu.hpp"
#include "bpftime_shm_internal.hpp"
#include "bpf_attach_ctx.hpp"
#include <cerrno>
#include <cstdlib>
#if __linux__
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#endif

#ifndef roundup
#define roundup(x, y)                                                          \
	({                                                                     \
		const decltype(y) __y = y;                                     \
		(((x) + (__y - 1)) / __y) * __y;                               \
	})
#endif // roundup

namespace bpftime
{

static size_t bpf_map_mmap_sz(unsigned int value_sz, unsigned int max_entries)
{
	const long page_sz = sysconf(_SC_PAGE_SIZE);
	size_t map_sz;

	map_sz = (size_t)roundup(value_sz, 8) * max_entries;
	map_sz = roundup(map_sz, page_sz);
	return map_sz;
}

void array_map_kernel_gpu_impl::init_map_fd()
{
	if (mmap_ptr != nullptr) {
		return;
	}
	if (kernel_map_id <= 0) {
		return;
	}
	map_fd = bpf_map_get_fd_by_id(kernel_map_id);
	if (map_fd < 0) {
		SPDLOG_ERROR("Failed to get fd for kernel map id {}",
			     kernel_map_id);
		return;
	}
	bpf_map_info info = {};
	unsigned int info_len = sizeof(info);
	int res = bpf_obj_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR("Failed to get info for kernel map id {}",
			     kernel_map_id);
	}
	_value_size = info.value_size;
	_max_entries = info.max_entries;
	value_data.resize(_value_size);
	SPDLOG_DEBUG(
		"create kernel user array map value size {}, max entries {}",
		_value_size, _max_entries);

	size_t mmap_sz = bpf_map_mmap_sz(_value_size, _max_entries);
	SPDLOG_DEBUG(
		"mmap shared array map, fd={}, mmap_sz={}, name={}, value_size={}, flags={}",
		map_fd, mmap_sz, info.name, info.value_size, info.map_flags);
	mmap_ptr = mmap(nullptr, mmap_sz, PROT_READ | PROT_WRITE,
			MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (mmap_ptr == MAP_FAILED) {
		SPDLOG_ERROR("Failed to mmap for kernel map id {}, err={}",
			     kernel_map_id, errno);
		return;
	}
	// What does this piece of code do?
	int prot;
	if (info.map_flags & BPF_F_RDONLY_PROG)
		prot = PROT_READ;
	else
		prot = PROT_READ | PROT_WRITE;
	void *mmaped = mmap(mmap_ptr, mmap_sz, prot, MAP_SHARED | MAP_FIXED,
			    map_fd, 0);
	if (mmaped == MAP_FAILED) {
		SPDLOG_ERROR(
			"Failed to re-mmap for kernel map id {}, err={}, prot={}",
			kernel_map_id, errno, prot);
		return;
	}
	mmap_ptr = mmaped;
}

array_map_kernel_gpu_impl::array_map_kernel_gpu_impl(
	boost::interprocess::managed_shared_memory &memory, int km_id,
	uint32_t value_size, uint32_t max_entries)
	: value_data(1, memory.get_segment_manager()),
	  _value_size(value_size),
	  _max_entries(max_entries),
	  kernel_map_id(km_id),
	  mmap_ptr(nullptr)
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
	  ,
	  agent_gpu_shared_mem(memory.get_segment_manager())
#endif

{
	if (kernel_map_id <= 0) {
		value_data.resize(static_cast<size_t>(_value_size) *
				  static_cast<size_t>(_max_entries));
	}
}

void *array_map_kernel_gpu_impl::elem_lookup(const void *key)
{
	if (kernel_map_id <= 0) {
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return nullptr;
		}
		return &value_data[key_val * _value_size];
	}
	if (map_fd < 0) {
		init_map_fd();
	}
	SPDLOG_DEBUG("Run lookup of shared array map, key={:x}",
		     (uintptr_t)key);
	if (mmap_ptr != nullptr) {
		auto key_val = *(uint32_t *)key;

		SPDLOG_DEBUG("mmap handled, key={}", key_val);
		if (key_val >= _max_entries) {
			errno = ENOENT;
			SPDLOG_DEBUG("Returned ENOENT");
			return nullptr;
		}
		auto result = &((uint8_t *)mmap_ptr)[key_val * _value_size];
		SPDLOG_DEBUG("Returned value addr: {:x}", (uintptr_t)result);
		return result;
	}
	// fallback to read kernel maps
	int res = bpf_map_lookup_elem(map_fd, key, value_data.data());
	if (res < 0) {
		return nullptr;
	}
	return (void *)value_data.data();
}

long array_map_kernel_gpu_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	if (kernel_map_id <= 0) {
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::copy((uint8_t *)value, (uint8_t *)value + _value_size,
			  &value_data[key_val * _value_size]);
		return 0;
	}
	if (map_fd < 0) {
		init_map_fd();
	}
	if (mmap_ptr != nullptr) {
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::copy((uint8_t *)value, (uint8_t *)value + _value_size,
			  &((uint8_t *)mmap_ptr)[key_val * _value_size]);
		return 0;
	}
	// fallback to read kernel maps
	return bpf_map_update_elem(map_fd, key, value, flags);
}

long array_map_kernel_gpu_impl::elem_delete(const void *key)
{
	if (kernel_map_id <= 0) {
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		memset(&value_data[key_val * _value_size], 0, _value_size);
		return 0;
	}
	if (map_fd < 0) {
		init_map_fd();
	}
	if (mmap_ptr != nullptr) {
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		memset(&((uint8_t *)mmap_ptr)[key_val * _value_size], 0,
		       _value_size);
		return 0;
	}
	// fallback to read kernel maps
	return bpf_map_delete_elem(map_fd, key);
}

int array_map_kernel_gpu_impl::map_get_next_key(const void *key, void *next_key)
{
	// Not found
	if (key == nullptr || *(uint32_t *)key >= _max_entries) {
		*(uint32_t *)next_key = 0;
		return 0;
	}
	uint32_t deref_key = *(uint32_t *)key;
	// Last element
	if (deref_key == _max_entries - 1) {
		errno = ENOENT;
		return -1;
	}
	auto key_val = *(uint32_t *)key;
	*(uint32_t *)next_key = key_val + 1;
	return 0;
}

array_map_kernel_gpu_impl::~array_map_kernel_gpu_impl()
{
	if (map_fd >= 0) {
		close(map_fd);
	}
}

static void *registered_host_memory = nullptr;

static void atexit_fn()
{
	SPDLOG_INFO("Calling cuMemHostUnregister");

	cuMemHostUnregister(registered_host_memory);
}

CUdeviceptr
array_map_kernel_gpu_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (kernel_map_id <= 0) {
		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr != agent_gpu_shared_mem.end()) {
			return itr->second;
		}
		const auto comm_host_base = reinterpret_cast<uintptr_t>(
			shm_holder.global_shared_memory.get_cuda_comm_shared_mem());
		const auto comm_device_base =
			bpftime::cuda::get_cuda_shared_mem_device_pointer();
		if (comm_host_base == 0 || comm_device_base == 0) {
			throw std::runtime_error(
				"CUDA shared memory device pointer not initialized");
		}
		const auto host_ptr =
			reinterpret_cast<uintptr_t>(value_data.data());
		const auto offset = static_cast<intptr_t>(host_ptr) -
				    static_cast<intptr_t>(comm_host_base);
		CUdeviceptr dev_ptr = static_cast<CUdeviceptr>(
			static_cast<uintptr_t>(
				static_cast<intptr_t>(comm_device_base) + offset));
		agent_gpu_shared_mem[pid] = dev_ptr;
		return dev_ptr;
	}
	if (map_fd < 0) {
		init_map_fd();
	}
	int pid = getpid();
	if (auto itr = agent_gpu_shared_mem.find(pid);
	    itr == agent_gpu_shared_mem.end()) {
		auto size = bpf_map_mmap_sz(_value_size, _max_entries);

		SPDLOG_INFO(
			"Initializing array_map_kernel_gpu_impl at pid {}, mmap_ptr = 0x{:x}, mmap_sz = {}",
			pid, (uintptr_t)mmap_ptr, size);
		registered_host_memory = mmap_ptr;
		if (auto err = cuMemHostRegister(
			    mmap_ptr, size,
			    CU_MEMHOSTREGISTER_PORTABLE |
				    CU_MEMHOSTREGISTER_DEVICEMAP);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR(
				"Unable to register host memory for kernel-gpu-shared map: {}",
				(int)err);

			throw std::runtime_error(
				"Unable to register host memory");
		}
		CUdeviceptr dev_ptr = 0;
		if (auto err =
			    cuMemHostGetDevicePointer(&dev_ptr, mmap_ptr, 0);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR(
				"cuMemHostGetDevicePointer failed for kernel-gpu-shared map host_ptr={:x}: {}",
				(uintptr_t)mmap_ptr, (int)err);
			throw std::runtime_error(
				"Unable to get device pointer for kernel-gpu-shared map");
		}
		agent_gpu_shared_mem[pid] = dev_ptr;
		atexit(atexit_fn);
		return dev_ptr;
	} else {
		return itr->second;
	}
}

} // namespace bpftime
