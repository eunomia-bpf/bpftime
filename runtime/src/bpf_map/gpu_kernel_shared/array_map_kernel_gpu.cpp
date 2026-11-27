/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "array_map_kernel_gpu.hpp"
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#if __linux__
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <sys/mman.h>
#include <unistd.h>
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

namespace
{

struct kernel_gpu_array_local_entry {
	int map_fd;
	void *mmap_ptr;
	size_t mmap_sz;
	uint32_t value_size;
	uint32_t max_entries;
	CUdeviceptr dev_ptr;
	bool dev_ptr_ready;
	bool host_registered;
};

shm_local_table<kernel_gpu_array_local_entry> g_kernel_gpu_local_table{};
static_assert(
	std::is_trivially_destructible_v<decltype(g_kernel_gpu_local_table)>);

#if __linux__
size_t bpf_map_mmap_sz(unsigned int value_sz, unsigned int max_entries)
{
	const long page_sz = sysconf(_SC_PAGE_SIZE);
	size_t map_sz = (size_t)roundup(value_sz, 8) * max_entries;
	map_sz = roundup(map_sz, page_sz);
	return map_sz;
}
#endif

void cleanup_kernel_gpu_local(kernel_gpu_array_local_entry &entry) noexcept
{
	if (entry.host_registered && entry.mmap_ptr != nullptr) {
		cuMemHostUnregister(entry.mmap_ptr);
		entry.host_registered = false;
	}
#if __linux__
	if (entry.mmap_ptr != nullptr && entry.mmap_sz > 0) {
		munmap(entry.mmap_ptr, entry.mmap_sz);
		entry.mmap_ptr = nullptr;
		entry.mmap_sz = 0;
	}
	if (entry.map_fd >= 0) {
		close(entry.map_fd);
		entry.map_fd = -1;
	}
#endif
	entry.dev_ptr = 0;
	entry.dev_ptr_ready = false;
}

#if __linux__
kernel_gpu_array_local_entry *resolve_kernel_gpu_local(shm_local_key key,
						       int kernel_map_id,
						       uint32_t fallback_vs,
						       uint32_t fallback_me)
{
	return g_kernel_gpu_local_table.get_or_init(
		key,
		[&](kernel_gpu_array_local_entry &out) -> bool {
			int fd = bpf_map_get_fd_by_id(kernel_map_id);
			if (fd < 0) {
				const int saved_errno = errno;
				SPDLOG_ERROR(
					"Failed to get fd for kernel map id {}",
					kernel_map_id);
				errno = saved_errno;
				return false;
			}
			out.map_fd = fd;
			out.mmap_ptr = nullptr;
			out.mmap_sz = 0;
			out.value_size = fallback_vs;
			out.max_entries = fallback_me;
			out.dev_ptr = 0;
			out.dev_ptr_ready = false;
			out.host_registered = false;

			bpf_map_info info = {};
			unsigned int info_len = sizeof(info);
			if (bpf_obj_get_info_by_fd(fd, &info, &info_len) < 0) {
				SPDLOG_ERROR(
					"Failed to get info for kernel map id {}",
					kernel_map_id);
				return true;
			}
			out.value_size = info.value_size;
			out.max_entries = info.max_entries;

			size_t mmap_sz =
				bpf_map_mmap_sz(out.value_size, out.max_entries);
			void *reserved = mmap(nullptr, mmap_sz,
					      PROT_READ | PROT_WRITE,
					      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
			if (reserved == MAP_FAILED) {
				SPDLOG_ERROR(
					"Failed to mmap for kernel map id {}, err={}",
					kernel_map_id, errno);
				return true;
			}
			int prot = (info.map_flags & BPF_F_RDONLY_PROG) ?
					   PROT_READ :
					   (PROT_READ | PROT_WRITE);
			void *mmaped = mmap(reserved, mmap_sz, prot,
					    MAP_SHARED | MAP_FIXED, fd, 0);
			if (mmaped == MAP_FAILED) {
				SPDLOG_ERROR(
					"Failed to re-mmap for kernel map id {}, err={}, prot={}",
					kernel_map_id, errno, prot);
				munmap(reserved, mmap_sz);
				return true;
			}
			out.mmap_ptr = mmaped;
			out.mmap_sz = mmap_sz;
			return true;
		},
		cleanup_kernel_gpu_local);
}
#endif

} // namespace

array_map_kernel_gpu_impl::array_map_kernel_gpu_impl(const shm_init_ctx &ctx,
						     int km_id,
						     uint32_t value_size,
						     uint32_t max_entries)
	: value_data(1, ctx.segment_manager()), _value_size(value_size),
	  _max_entries(max_entries), kernel_map_id(km_id),
	  has_kernel_map(km_id > 0)
{
	if (!has_kernel_map) {
		value_data.resize(static_cast<size_t>(_value_size) *
				  static_cast<size_t>(_max_entries));
		local_key_ = ctx.allocate_local_key();
		return;
	}
#if __linux__
	if (_value_size == 0 || _max_entries == 0) {
		int fd = bpf_map_get_fd_by_id(kernel_map_id);
		if (fd >= 0) {
			bpf_map_info info = {};
			unsigned int info_len = sizeof(info);
			if (bpf_obj_get_info_by_fd(fd, &info, &info_len) == 0) {
				_value_size = info.value_size;
				_max_entries = info.max_entries;
			}
			close(fd);
		}
	}
#endif
	if (_value_size > 0) {
		value_data.resize(_value_size);
	}
	local_key_ = ctx.allocate_local_key();
}

void *array_map_kernel_gpu_impl::elem_lookup(const void *key)
{
	if (!has_kernel_map) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return nullptr;
		}
		return &value_data[static_cast<size_t>(key_val) * _value_size];
	}
#if __linux__
	auto *local = resolve_kernel_gpu_local(local_key_, kernel_map_id,
					       _value_size, _max_entries);
	if (local == nullptr)
		return nullptr;
	if (local->mmap_ptr != nullptr) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= local->max_entries) {
			errno = ENOENT;
			return nullptr;
		}
		return &static_cast<uint8_t *>(local->mmap_ptr)
				[static_cast<size_t>(key_val) *
				 local->value_size];
	}
	if (value_data.size() < local->value_size)
		return nullptr;
	int res = bpf_map_lookup_elem(local->map_fd, key, value_data.data());
	if (res < 0)
		return nullptr;
	return value_data.data();
#else
	errno = ENOTSUP;
	return nullptr;
#endif
}

long array_map_kernel_gpu_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	if (!has_kernel_map) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::copy((const uint8_t *)value,
			  (const uint8_t *)value + _value_size,
			  &value_data[static_cast<size_t>(key_val) *
				      _value_size]);
		return 0;
	}
#if __linux__
	auto *local = resolve_kernel_gpu_local(local_key_, kernel_map_id,
					       _value_size, _max_entries);
	if (local == nullptr)
		return -1;
	if (local->mmap_ptr != nullptr) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= local->max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::copy((const uint8_t *)value,
			  (const uint8_t *)value + local->value_size,
			  &static_cast<uint8_t *>(local->mmap_ptr)
				   [static_cast<size_t>(key_val) *
				    local->value_size]);
		return 0;
	}
	return bpf_map_update_elem(local->map_fd, key, value, flags);
#else
	errno = ENOTSUP;
	return -1;
#endif
}

long array_map_kernel_gpu_impl::elem_delete(const void *key)
{
	if (!has_kernel_map) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		memset(&value_data[static_cast<size_t>(key_val) * _value_size],
		       0, _value_size);
		return 0;
	}
#if __linux__
	auto *local = resolve_kernel_gpu_local(local_key_, kernel_map_id,
					       _value_size, _max_entries);
	if (local == nullptr)
		return -1;
	if (local->mmap_ptr != nullptr) {
		auto key_val = *(const uint32_t *)key;
		if (key_val >= local->max_entries) {
			errno = ENOENT;
			return -1;
		}
		memset(&static_cast<uint8_t *>(local->mmap_ptr)
				[static_cast<size_t>(key_val) *
				 local->value_size],
		       0, local->value_size);
		return 0;
	}
	return bpf_map_delete_elem(local->map_fd, key);
#else
	errno = ENOTSUP;
	return -1;
#endif
}

int array_map_kernel_gpu_impl::map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr || *(const uint32_t *)key >= _max_entries) {
		*(uint32_t *)next_key = 0;
		return 0;
	}
	uint32_t deref_key = *(const uint32_t *)key;
	if (deref_key == _max_entries - 1) {
		errno = ENOENT;
		return -1;
	}
	*(uint32_t *)next_key = deref_key + 1;
	return 0;
}

array_map_kernel_gpu_impl::~array_map_kernel_gpu_impl()
{
	g_kernel_gpu_local_table.evict(local_key_, cleanup_kernel_gpu_local);
	release_local_key(value_data.get_allocator().get_segment_manager(),
			  local_key_);
}

CUdeviceptr
array_map_kernel_gpu_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (!has_kernel_map) {
		auto *local = g_kernel_gpu_local_table.get_or_init(
			local_key_,
			[&](kernel_gpu_array_local_entry &out) -> bool {
				out.map_fd = -1;
				out.mmap_ptr = nullptr;
				out.mmap_sz = 0;
				out.value_size = _value_size;
				out.max_entries = _max_entries;
				out.host_registered = false;
				const auto comm_host_base = reinterpret_cast<
					uintptr_t>(
					shm_holder.global_shared_memory
						.get_cuda_comm_shared_mem());
				const auto comm_device_base = bpftime::cuda::
					get_cuda_shared_mem_device_pointer();
				if (comm_host_base == 0 ||
				    comm_device_base == 0)
					return false;
				const auto host_ptr = reinterpret_cast<
					uintptr_t>(value_data.data());
				const auto offset =
					static_cast<intptr_t>(host_ptr) -
					static_cast<intptr_t>(comm_host_base);
				out.dev_ptr = static_cast<CUdeviceptr>(
					static_cast<uintptr_t>(
						static_cast<intptr_t>(
							comm_device_base) +
						offset));
				out.dev_ptr_ready = true;
				return true;
			});
		if (local == nullptr || !local->dev_ptr_ready) {
			throw std::runtime_error(
				"CUDA shared memory device pointer not initialized");
		}
		return local->dev_ptr;
	}
#if __linux__
	auto *local = resolve_kernel_gpu_local(local_key_, kernel_map_id,
					       _value_size, _max_entries);
	if (local == nullptr || local->mmap_ptr == nullptr) {
		throw std::runtime_error(
			"Kernel map is not mmap-able (required for kernel-gpu-shared map)");
	}
	if (!local->dev_ptr_ready) {
		auto size =
			bpf_map_mmap_sz(local->value_size, local->max_entries);
		if (auto err = cuMemHostRegister(
			    local->mmap_ptr, size,
			    CU_MEMHOSTREGISTER_PORTABLE |
				    CU_MEMHOSTREGISTER_DEVICEMAP);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR(
				"Unable to register host memory for kernel-gpu-shared map: {}",
				(int)err);
			throw std::runtime_error(
				"Unable to register host memory");
		}
		local->host_registered = true;
		CUdeviceptr dev_ptr = 0;
		if (auto err = cuMemHostGetDevicePointer(&dev_ptr,
							 local->mmap_ptr, 0);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR(
				"cuMemHostGetDevicePointer failed for kernel-gpu-shared map host_ptr={:x}: {}",
				(uintptr_t)local->mmap_ptr, (int)err);
			throw std::runtime_error(
				"Unable to get device pointer for kernel-gpu-shared map");
		}
		local->dev_ptr = dev_ptr;
		local->dev_ptr_ready = true;
	}
	return local->dev_ptr;
#else
	throw std::runtime_error("Kernel-GPU shared map requires Linux");
#endif
}

} // namespace bpftime
