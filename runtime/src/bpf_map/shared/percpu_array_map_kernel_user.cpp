/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/shared/percpu_array_map_kernel_user.hpp"
#include "spdlog/spdlog.h"
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <cerrno>
#include <unistd.h>

namespace bpftime
{

namespace
{

struct kernel_percpu_local_entry {
	int map_fd;
};

shm_local_table<kernel_percpu_local_entry> g_kernel_percpu_local_table{};
static_assert(
	std::is_trivially_destructible_v<decltype(g_kernel_percpu_local_table)>);

size_t percpu_elem_stride(uint32_t value_size) noexcept
{
	return (static_cast<size_t>(value_size) + 7u) & ~static_cast<size_t>(7u);
}

void cleanup_kernel_percpu_local(kernel_percpu_local_entry &entry) noexcept
{
	if (entry.map_fd >= 0) {
		close(entry.map_fd);
		entry.map_fd = -1;
	}
}

int resolve_kernel_percpu_fd(shm_local_key key, int kernel_map_id)
{
	const auto *local = g_kernel_percpu_local_table.get_or_init(
		key,
		[&](kernel_percpu_local_entry &out) -> bool {
			int fd = bpf_map_get_fd_by_id(kernel_map_id);
			if (fd < 0) {
				const int saved_errno = errno;
				SPDLOG_ERROR(
					"Unable to open kernel map id {}: {}",
					kernel_map_id, saved_errno);
				errno = saved_errno;
				return false;
			}
			out.map_fd = fd;
			return true;
		},
		cleanup_kernel_percpu_local);
	return local != nullptr ? local->map_fd : -1;
}

} // namespace

void *percpu_array_map_kernel_user_impl::elem_lookup(const void *key)
{
	if (_value_size == 0 || ncpu <= 0) {
		errno = EINVAL;
		return nullptr;
	}
	const size_t needed =
		percpu_elem_stride(_value_size) * static_cast<size_t>(ncpu);
	if (value_data.size() < needed) {
		errno = EINVAL;
		return nullptr;
	}
	int fd = resolve_kernel_percpu_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return nullptr;
	if (bpf_map_lookup_elem(fd, key, value_data.data()) < 0)
		return nullptr;
	return value_data.data();
}

long percpu_array_map_kernel_user_impl::elem_update(const void *key,
						    const void *value,
						    uint64_t flags)
{
	int fd = resolve_kernel_percpu_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_update_elem(fd, key, value, flags);
}

long percpu_array_map_kernel_user_impl::elem_delete(const void *key)
{
	int fd = resolve_kernel_percpu_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_delete_elem(fd, key);
}

int percpu_array_map_kernel_user_impl::map_get_next_key(const void *key,
							void *next_key)
{
	int fd = resolve_kernel_percpu_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_get_next_key(fd, key, next_key);
}

percpu_array_map_kernel_user_impl::percpu_array_map_kernel_user_impl(
	const shm_init_ctx &ctx, int kernel_map_id)
	: kernel_map_id(kernel_map_id), value_data(1, ctx.segment_manager())
{
	int possible = libbpf_num_possible_cpus();
	ncpu = possible > 0 ? possible : 0;
	if (kernel_map_id > 0) {
		int fd = bpf_map_get_fd_by_id(kernel_map_id);
		if (fd >= 0) {
			bpf_map_info map_info{};
			uint32_t sz = sizeof(map_info);
			const int rc =
				bpf_obj_get_info_by_fd(fd, &map_info, &sz);
			close(fd);
			if (rc == 0) {
				_value_size = map_info.value_size;
				_max_entries = map_info.max_entries;
				if (_value_size > 0 && ncpu > 0) {
					value_data.resize(
						percpu_elem_stride(_value_size) *
						static_cast<size_t>(ncpu));
				}
			}
		}
		local_key_ = ctx.allocate_local_key();
	}
}

percpu_array_map_kernel_user_impl::~percpu_array_map_kernel_user_impl()
{
	g_kernel_percpu_local_table.evict(local_key_,
					  cleanup_kernel_percpu_local);
	release_local_key(value_data.get_allocator().get_segment_manager(),
			  local_key_);
}

} // namespace bpftime
