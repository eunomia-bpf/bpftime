/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf/bpf.h"
#include "spdlog/spdlog.h"
#include <bpf_map/shared/percpu_array_map_kernel_user.hpp>
#include <linux/bpf.h>
namespace bpftime
{
void percpu_array_map_kernel_user_impl::init_map_fd()
{
	kernel_map_fd = bpf_map_get_fd_by_id(kernel_map_id);
	if (kernel_map_fd < 0) {
		SPDLOG_ERROR(
			"Failed to retrive kernel map fd by kernel map id: {}",
			kernel_map_fd);
		return;
	}
	bpf_map_info map_info;
	uint32_t sz = sizeof(map_info);
	if (int err = bpf_obj_get_info_by_fd(kernel_map_fd, &map_info, &sz);
	    err < 0) {
		SPDLOG_ERROR(
			"Failed to get map detail of kernel map fd {}, err={}",
			kernel_map_fd, err);
		return;
	}
	_value_size = map_info.value_size;
	_max_entries = map_info.max_entries;
	value_data.resize(_value_size * ncpu);
}
void *percpu_array_map_kernel_user_impl::elem_lookup(const void *key)
{
	int err = bpf_map_lookup_elem(kernel_map_fd, key,
				      (void *)value_data.data());
	if (err < 0) {
		SPDLOG_ERROR(
			"Failed to perform elem lookup of percpu array {}",
			kernel_map_fd);
		return nullptr;
	}
	return value_data.data();
}

long percpu_array_map_kernel_user_impl::elem_update(const void *key,
						    const void *value,
						    uint64_t flags)
{
	return bpf_map_update_elem(kernel_map_fd, key, value, flags);
}

long percpu_array_map_kernel_user_impl::elem_delete(const void *key)
{
	return bpf_map_delete_elem(kernel_map_fd, key);
}

int percpu_array_map_kernel_user_impl::map_get_next_key(const void *key,
							void *next_key)
{
	return bpf_map_get_next_key(kernel_map_fd, key, next_key);
}
percpu_array_map_kernel_user_impl::percpu_array_map_kernel_user_impl(
	boost::interprocess::managed_shared_memory &memory, int kernel_map_id)
	: kernel_map_id(kernel_map_id),
	  value_data(1, memory.get_segment_manager())
{
	ncpu = sysconf(_SC_NPROCESSORS_ONLN);
}
percpu_array_map_kernel_user_impl::~percpu_array_map_kernel_user_impl()
{
}
} // namespace bpftime
