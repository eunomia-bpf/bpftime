/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PERF_EVENT_ARRAY_KERNEL_USER
#define _BPFTIME_PERF_EVENT_ARRAY_KERNEL_USER
#include <boost/interprocess/managed_shared_memory.hpp>
#if __linux__
#include <linux/perf_event.h>
#endif
#include <pthread.h>
#include <vector>

namespace bpftime
{

// Here is an implementation of a perf event array that can output data from
// both userspace and kernel space It's corresponded with a normal perf event
// array in kernel, and it will be used by kernel ebpf programs But from the
// userspace, it will hold a user ringbuf (provided by libbpf). Once data was
// written, it will commit the data to the ringbuf. A kernel program will be
// attached to an intervally triggered event (e.g a timer perf event). This
// program will examine if data was available in the user ringbuf, and writes
// the data into the corresponding kernel perf event array.

// Data definition of things from userspace to kernel through user rb
// [8 bytes of data length][data]
class perf_event_array_kernel_user_impl {
	uint32_t dummy = 0xffffffff;
	uint32_t max_ent;
	int kernel_perf_id;
	int user_rb_id;
	int pfd = -1;
	int transporter_prog_fd = -1;
    public:
	const static bool should_lock = false;
	perf_event_array_kernel_user_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t value_size, uint32_t max_entries,
		int kernel_perf_id);
	virtual ~perf_event_array_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	struct user_ringbuffer_wrapper *ensure_current_map_user_ringbuf();
	int output_data_into_kernel(const void *buf, size_t size);
	int get_user_ringbuf_fd();
};

// Create a bpf program that will check if the user_ringbuf has data and copy
// the data into the perf event
std::vector<uint64_t>
create_transporting_kernel_ebpf_program(int user_ringbuf_fd,
					int perf_event_array_fd);
// Create an intervally triggered perf event
int create_intervally_triggered_perf_event(int duration_ms);

} // namespace bpftime
#endif
