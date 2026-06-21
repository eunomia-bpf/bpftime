/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_USER_PROG_ARRAY_HPP
#define _BPFTIME_KERNEL_USER_PROG_ARRAY_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>

namespace bpftime
{

class prog_array_kernel_user_impl {
	int kernel_map_id = -1;
	int map_fd = -1;
	uint32_t _max_entries = 0;

	bool init_map_fd();

    public:
	const static bool should_lock = false;
	prog_array_kernel_user_impl(
		boost::interprocess::managed_shared_memory &memory, int km_id);
	~prog_array_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime

#endif // _BPFTIME_KERNEL_USER_PROG_ARRAY_HPP
