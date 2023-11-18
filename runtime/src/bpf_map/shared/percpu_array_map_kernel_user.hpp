/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PERCPU_ARRAY_MAP_KERNEL_USER_HPP
#define _BPFTIME_PERCPU_ARRAY_MAP_KERNEL_USER_HPP
#include <cstdint>
#include <bpf_map/map_common_def.hpp>
namespace bpftime
{

// This is a simple percpu array map implementation, which could be accessed from both userspace ebpf programs and kernel ebpf programs
// It just uses syscalls to operate the corresponding kernel map :(

class percpu_array_map_kernel_user_impl {
	uint32_t _value_size;
	uint32_t _max_entries;
	int kernel_map_fd = -1;
	int kernel_map_id;
    int ncpu;
	void init_map_fd();
	bytes_vec value_data;

    public:
	const static bool should_lock = false;
	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
	percpu_array_map_kernel_user_impl(
		boost::interprocess::managed_shared_memory &memory,
		int kernel_map_id);
	~percpu_array_map_kernel_user_impl();
};
} // namespace bpftime
#endif
