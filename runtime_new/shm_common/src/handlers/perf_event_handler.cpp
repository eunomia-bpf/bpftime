/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "handlers/handler_common_def.hpp"
#include "linux/perf_event.h"
#include "spdlog/spdlog.h"
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <cassert>
#include <cstring>
#include <handlers/perf_event_handler.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <unistd.h>
#include <spdlog/fmt/bin_to_hex.h>
#include <variant>

namespace bpftime
{
namespace shm_common
{
// attach to replace or filter self define types
bpf_perf_event_handler::bpf_perf_event_handler(
	int type, uint64_t offset, int pid, const char *module_name,
	boost::interprocess::managed_shared_memory &mem, bool default_enabled)
	: type(type), enabled(default_enabled),
	  private_data(uprobe_perf_event_private_data{
		  .offset = offset,
		  .pid = pid,
		  ._module_name = boost_shm_string(
			  char_allocator(mem.get_segment_manager())) })
{
	std::get<uprobe_perf_event_private_data>(this->private_data)
		._module_name = module_name;
}

// create uprobe/uretprobe with new perf event attr
bpf_perf_event_handler::bpf_perf_event_handler(
	bool is_retprobe, uint64_t offset, int pid, const char *module_name,
	size_t ref_ctr_off, boost::interprocess::managed_shared_memory &mem)
	: private_data(uprobe_perf_event_private_data{
		  .offset = offset,
		  .pid = pid,
		  .ref_ctr_off = ref_ctr_off,
		  ._module_name = boost_shm_string(
			  char_allocator(mem.get_segment_manager())) })
{
	if (is_retprobe) {
		type = (int)bpf_event_type::BPF_TYPE_URETPROBE;
	} else {
		type = (int)bpf_event_type::BPF_TYPE_UPROBE;
	}
	std::get<uprobe_perf_event_private_data>(this->private_data)
		._module_name = module_name;
	SPDLOG_INFO(
		"Created uprobe/uretprobe perf event handler, module name {}, offset {:x}",
		module_name, offset);
}

// create tracepoint
bpf_perf_event_handler::bpf_perf_event_handler(
	int pid, int32_t tracepoint_id,
	boost::interprocess::managed_shared_memory &mem)
	: type((int)bpf_event_type::PERF_TYPE_TRACEPOINT),
	  private_data(tracepoint_perf_event_private_data{
		  .pid = pid,
		  .tracepoint_id = tracepoint_id })
{
}

bpf_perf_event_handler::bpf_perf_event_handler(
	int cpu, int32_t sample_type, int64_t config,
	boost::interprocess::managed_shared_memory &mem)
	: type((int)bpf_event_type::PERF_TYPE_SOFTWARE),
	  private_data(software_perf_event_private_data{
		  .sw_perf = boost::interprocess::make_managed_shared_ptr(
			  mem.construct<software_perf_event_data>(
				  boost::interprocess::anonymous_instance)(
				  cpu, config, sample_type, mem),
			  mem) })

{
}

std::optional<software_perf_event_weak_ptr>
bpf_perf_event_handler::try_get_software_perf_data_weak_ptr() const
{
	if (std::holds_alternative<software_perf_event_private_data>(
		    private_data)) {
		return software_perf_event_weak_ptr(
			std::get<software_perf_event_private_data>(private_data)
				.sw_perf);
	} else {
		return {};
	}
}

std::optional<void *>
bpf_perf_event_handler::try_get_software_perf_data_raw_buffer(
	size_t buffer_size) const
{
	if (std::holds_alternative<software_perf_event_private_data>(
		    private_data)) {
		return std::get<software_perf_event_private_data>(private_data)
			.sw_perf->ensure_mmap_buffer(buffer_size);
	} else {
		return {};
	}
}

int bpf_perf_event_handler::enable() const
{
	enabled = true;
	// TODO: implement enable logic.
	// If This is a server, should inject the agent into the target
	// process.

	SPDLOG_DEBUG("Enabling perf event for module name: {}, offset {:x}",
		     _module_name.c_str(), offset);
	return 0;
}
int bpf_perf_event_handler::disable() const
{
	SPDLOG_DEBUG("Disabling perf event for module name: {}, offset {:x}",
		     _module_name.c_str(), offset);
	enabled = false;
	return 0;
}
} // namespace shm_common
} // namespace bpftime
