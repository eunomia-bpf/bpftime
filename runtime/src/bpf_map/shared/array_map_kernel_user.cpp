/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/shared/array_map_kernel_user.hpp>
#include <cerrno>
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

struct kernel_array_local_entry {
	int map_fd;
	void *mmap_ptr;
	size_t mmap_sz;
	uint32_t value_size;
	uint32_t max_entries;
};

shm_local_table<kernel_array_local_entry> g_kernel_array_local_table{};
static_assert(
	std::is_trivially_destructible_v<decltype(g_kernel_array_local_table)>);

#if __linux__
size_t bpf_map_mmap_sz(unsigned int value_sz, unsigned int max_entries)
{
	const long page_sz = sysconf(_SC_PAGE_SIZE);
	size_t map_sz = (size_t)roundup(value_sz, 8) * max_entries;
	map_sz = roundup(map_sz, page_sz);
	return map_sz;
}

void cleanup_kernel_array_local(kernel_array_local_entry &entry) noexcept
{
	if (entry.mmap_ptr != nullptr && entry.mmap_sz > 0) {
		munmap(entry.mmap_ptr, entry.mmap_sz);
		entry.mmap_ptr = nullptr;
		entry.mmap_sz = 0;
	}
	if (entry.map_fd >= 0) {
		close(entry.map_fd);
		entry.map_fd = -1;
	}
}

const kernel_array_local_entry *resolve_kernel_array_local(shm_local_key key,
							   int kernel_map_id,
							   uint32_t fallback_vs,
							   uint32_t fallback_me)
{
	return g_kernel_array_local_table.get_or_init(
		key,
		[&](kernel_array_local_entry &out) -> bool {
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
		cleanup_kernel_array_local);
}
#endif

} // namespace

array_map_kernel_user_impl::array_map_kernel_user_impl(const shm_init_ctx &ctx,
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
	if (_value_size > 0)
		value_data.resize(_value_size);
	local_key_ = ctx.allocate_local_key();
}

void *array_map_kernel_user_impl::elem_lookup(const void *key)
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
	const auto *local = resolve_kernel_array_local(
		local_key_, kernel_map_id, _value_size, _max_entries);
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
	if (local->value_size == 0 || value_data.size() < local->value_size)
		return nullptr;
	if (bpf_map_lookup_elem(local->map_fd, key, value_data.data()) < 0)
		return nullptr;
	return value_data.data();
#else
	errno = ENOTSUP;
	return nullptr;
#endif
}

long array_map_kernel_user_impl::elem_update(const void *key, const void *value,
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
	const auto *local = resolve_kernel_array_local(
		local_key_, kernel_map_id, _value_size, _max_entries);
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

long array_map_kernel_user_impl::elem_delete(const void *key)
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
	const auto *local = resolve_kernel_array_local(
		local_key_, kernel_map_id, _value_size, _max_entries);
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

int array_map_kernel_user_impl::map_get_next_key(const void *key,
						 void *next_key)
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

void *array_map_kernel_user_impl::get_raw_data() const
{
	if (!has_kernel_map)
		return (void *)value_data.data();
#if __linux__
	const auto *local = g_kernel_array_local_table.find(local_key_);
	if (local != nullptr && local->mmap_ptr != nullptr)
		return local->mmap_ptr;
#endif
	return (void *)value_data.data();
}

array_map_kernel_user_impl::~array_map_kernel_user_impl()
{
#if __linux__
	g_kernel_array_local_table.evict(local_key_,
					 cleanup_kernel_array_local);
#endif
	release_local_key(value_data.get_allocator().get_segment_manager(),
			  local_key_);
}

} // namespace bpftime
