/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _PERF_EVENT_HANDLER
#define _PERF_EVENT_HANDLER
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <cstddef>
#include <optional>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
#include <variant>
#include <handlers/handler_common_def.hpp>
#include "software_perf_event_data/software_perf_event_data.hpp"

namespace bpftime
{

namespace shm_common
{
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

// Private data for a uprobe type perf event instance
struct uprobe_perf_event_private_data {
	// Offset of the hooked function
	uint64_t offset;
	// Target pid
	int pid;
	size_t ref_ctr_off = 0;
	boost_shm_string _module_name;
};

// Private data for a tracepoint type perf event instance
struct tracepoint_perf_event_private_data {
	int pid;
	// Tracepoint id at /sys/kernel/tracing/events/syscalls/*/id, used to
	// indicate which syscall to trace
	int32_t tracepoint_id = -1;
};
// Private data for a software perf event instance, used for PERF_EVENT_MAP
struct software_perf_event_private_data {
	// Things needed by software perf event
	software_perf_event_shared_ptr sw_perf;
};

using perf_event_private_data_variant =
	std::variant<uprobe_perf_event_private_data,
		     tracepoint_perf_event_private_data,
		     software_perf_event_private_data>;

// perf event handler
struct bpf_perf_event_handler {
	// Type of the perf event instance
	int type;
	// Enabled?
	mutable bool enabled = false;
	int enable() const;
	int disable() const;
	// Private data of different perf event types
	perf_event_private_data_variant private_data;

	// Assume this is a software perf event instance, and get the weak ptr
	// to the private data. Return none if assumption failed
	std::optional<software_perf_event_weak_ptr>
	try_get_software_perf_data_weak_ptr() const;

	// Assume this is a software perf event instance, and get the buffer to
	// the inner data. Return none if assumption failed
	std::optional<void *>
	try_get_software_perf_data_raw_buffer(size_t buffer_size) const;

	// Construct this perf event instance, filling offset, pid, and module
	// name
	bpf_perf_event_handler(int type, uint64_t offset, int pid,
			       const char *module_name,
			       boost::interprocess::managed_shared_memory &mem,
			       bool default_enabled = false);

	// Construct this perf event instance with uprobe or uretprobe data
	bpf_perf_event_handler(bool is_retprobe, uint64_t offset, int pid,
			       const char *module_name, size_t ref_ctr_off,
			       boost::interprocess::managed_shared_memory &mem);

	// Construct this perf event instance with tracepoint type
	bpf_perf_event_handler(int pid, int32_t tracepoint_id,
			       boost::interprocess::managed_shared_memory &mem);
	// Construct this perf event instance with software type. In this case,
	// this perf event instance may not be used as an attach target
	bpf_perf_event_handler(int cpu, int32_t sample_type, int64_t config,
			       boost::interprocess::managed_shared_memory &mem);
};
} // namespace shm_common

} // namespace bpftime

#endif
