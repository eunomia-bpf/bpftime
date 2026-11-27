/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PERF_EVENT_ARRAY_KERNEL_USER
#define _BPFTIME_PERF_EVENT_ARRAY_KERNEL_USER
#include "bpftime_shm_safe.hpp"
#if __linux__
#include <linux/perf_event.h>
#endif
#include <pthread.h>
#include <vector>

namespace bpftime
{

class perf_event_array_kernel_user_impl {
	uint32_t dummy = 0xffffffff;
	uint32_t max_ent;
	int kernel_perf_id;
	int user_rb_id = -1;
	shm_local_key local_key_{};
	boost::interprocess::allocator<
		char, boost::interprocess::managed_shared_memory::segment_manager>
		seg_alloc_;

    public:
	const static bool should_lock = false;
	perf_event_array_kernel_user_impl(const shm_init_ctx &ctx,
					  uint32_t key_size,
					  uint32_t value_size,
					  uint32_t max_entries,
					  int kernel_perf_id);
	~perf_event_array_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	int output_data_into_kernel(const void *buf, size_t size);
	int get_user_ringbuf_fd();
};

std::vector<uint64_t>
create_transporting_kernel_ebpf_program(int user_ringbuf_fd,
					int perf_event_array_fd);
int create_intervally_triggered_perf_event(int duration_ms);

} // namespace bpftime
#endif
