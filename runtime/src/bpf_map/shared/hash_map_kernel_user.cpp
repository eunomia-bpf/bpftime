/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <bpf_map/shared/hash_map_kernel_user.hpp>
#include <cerrno>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

namespace bpftime
{

namespace
{

struct kernel_hash_local_entry {
	int map_fd;
};

shm_local_table<kernel_hash_local_entry> g_kernel_hash_local_table{};
static_assert(
	std::is_trivially_destructible_v<decltype(g_kernel_hash_local_table)>);

void cleanup_kernel_hash_local(kernel_hash_local_entry &entry) noexcept
{
	if (entry.map_fd >= 0) {
		close(entry.map_fd);
		entry.map_fd = -1;
	}
}

int resolve_kernel_hash_fd(shm_local_key key, int kernel_map_id)
{
	const auto *local = g_kernel_hash_local_table.get_or_init(
		key,
		[&](kernel_hash_local_entry &out) -> bool {
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
			return true;
		},
		cleanup_kernel_hash_local);
	return local != nullptr ? local->map_fd : -1;
}

} // namespace

hash_map_kernel_user_impl::hash_map_kernel_user_impl(const shm_init_ctx &ctx,
						     int km_id)
	: key_vec(1, ctx.segment_manager()),
	  value_vec(1, ctx.segment_manager()), kernel_map_id(km_id)
{
	if (kernel_map_id > 0) {
		int fd = bpf_map_get_fd_by_id(kernel_map_id);
		if (fd >= 0) {
			bpf_map_info info = {};
			unsigned int info_len = sizeof(info);
			const int rc =
				bpf_obj_get_info_by_fd(fd, &info, &info_len);
			close(fd);
			if (rc == 0) {
				_key_size = info.key_size;
				_value_size = info.value_size;
				if (_key_size > 0)
					key_vec.resize(_key_size);
				if (_value_size > 0)
					value_vec.resize(_value_size);
			}
		}
		local_key_ = ctx.allocate_local_key();
	}
}

void *hash_map_kernel_user_impl::elem_lookup(const void *key)
{
	int fd = resolve_kernel_hash_fd(local_key_, kernel_map_id);
	if (fd < 0 || _value_size == 0 || value_vec.size() < _value_size)
		return nullptr;
	if (bpf_map_lookup_elem(fd, key, value_vec.data()) < 0)
		return nullptr;
	return value_vec.data();
}

long hash_map_kernel_user_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	int fd = resolve_kernel_hash_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_update_elem(fd, key, value, flags);
}

long hash_map_kernel_user_impl::elem_delete(const void *key)
{
	int fd = resolve_kernel_hash_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_delete_elem(fd, key);
}

int hash_map_kernel_user_impl::map_get_next_key(const void *key, void *next_key)
{
	int fd = resolve_kernel_hash_fd(local_key_, kernel_map_id);
	if (fd < 0)
		return -1;
	return bpf_map_get_next_key(fd, key, next_key);
}

hash_map_kernel_user_impl::~hash_map_kernel_user_impl()
{
	g_kernel_hash_local_table.evict(local_key_, cleanup_kernel_hash_local);
	release_local_key(key_vec.get_allocator().get_segment_manager(),
			  local_key_);
}

} // namespace bpftime
