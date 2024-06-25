/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <bpf_map/shared/hash_map_kernel_user.hpp>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

namespace bpftime
{

void hash_map_kernel_user_impl::init_map_fd()
{
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
	_key_size = info.key_size;
	_value_size = info.value_size;
	SPDLOG_DEBUG("create kernel user hash map key size {}, value size {}",
		      _key_size, _value_size);
	key_vec.resize(_key_size);
	value_vec.resize(_value_size);
}

hash_map_kernel_user_impl::hash_map_kernel_user_impl(
	managed_shared_memory &memory, int km_id)
	: key_vec(1, memory.get_segment_manager()),
	  value_vec(1, memory.get_segment_manager()), kernel_map_id(km_id)
{
}

void *hash_map_kernel_user_impl::elem_lookup(const void *key)
{
	SPDLOG_TRACE("Peform elem lookup of hash map");
	if (map_fd < 0) {
		init_map_fd();
	}
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	int res = bpf_map_lookup_elem(map_fd, key, value_vec.data());
	if (res < 0) {
		return nullptr;
	}
	return value_vec.data();
}

long hash_map_kernel_user_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	SPDLOG_TRACE("Peform elem update of hash map");
	if (map_fd < 0) {
		init_map_fd();
	}
	SPDLOG_DEBUG("Update shared hash map");
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_update_elem(map_fd, key, value, flags);
}

long hash_map_kernel_user_impl::elem_delete(const void *key)
{
	SPDLOG_TRACE("Peform elem delete of hash map");
	if (map_fd < 0) {
		init_map_fd();
	}
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_delete_elem(map_fd, key);
}

int hash_map_kernel_user_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_TRACE("Peform get next key of hash map");
	if (map_fd < 0) {
		init_map_fd();
	}
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_get_next_key(map_fd, key, next_key);
}

hash_map_kernel_user_impl::~hash_map_kernel_user_impl()
{
	if (map_fd >= 0) {
		close(map_fd);
	}
}

} // namespace bpftime
