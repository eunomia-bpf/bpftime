/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "linux/perf_event.h"
#include "spdlog/spdlog.h"
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <cassert>
#include <cstring>
#include <handler/perf_event_handler.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <unistd.h>
#include <spdlog/fmt/bin_to_hex.h>

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#define READ_ONCE_U64(x) (*(volatile uint64_t *)&x)
#define WRITE_ONCE_U64(x, v) (*(volatile uint64_t *)&x) = (v)

#define barrier() asm volatile("" ::: "memory")
#ifdef __x86_64__
#define smp_store_release_u64(p, v)                                            \
	do {                                                                   \
		barrier();                                                     \
		WRITE_ONCE_U64(*p, v);                                         \
	} while (0)

#define smp_load_acquire_u64(p)                                                \
	({                                                                     \
		uint64_t ___p = READ_ONCE_U64(*p);                             \
		barrier();                                                     \
		___p;                                                          \
	})

#else
#error Only supports x86_64
#endif

namespace bpftime
{
// attach to replace or filter self define types
bpf_perf_event_handler::bpf_perf_event_handler(
	bpf_event_type type, uint64_t offset, int pid, const char *module_name,
	boost::interprocess::managed_shared_memory &mem, bool default_enabled)
	: type(type), enabled(default_enabled), offset(offset), pid(pid),
	  _module_name(char_allocator(mem.get_segment_manager()))
{
	this->_module_name = module_name;
}

// create uprobe/uretprobe with new perf event attr
bpf_perf_event_handler::bpf_perf_event_handler(
	bool is_retprobe, uint64_t offset, int pid, const char *module_name,
	size_t ref_ctr_off, boost::interprocess::managed_shared_memory &mem)
	: offset(offset), pid(pid), ref_ctr_off(ref_ctr_off),
	  _module_name(char_allocator(mem.get_segment_manager()))
{
	if (is_retprobe) {
		type = bpf_event_type::BPF_TYPE_URETPROBE;
	} else {
		type = bpf_event_type::BPF_TYPE_UPROBE;
	}
	this->_module_name = module_name;
	spdlog::info(
		"Created uprobe/uretprobe perf event handler, module name {}, offset {:x}",
		module_name, offset);
}

// create tracepoint
bpf_perf_event_handler::bpf_perf_event_handler(
	int pid, int32_t tracepoint_id,
	boost::interprocess::managed_shared_memory &mem)
	: type(bpf_event_type::PERF_TYPE_TRACEPOINT), pid(pid),
	  _module_name(char_allocator(mem.get_segment_manager())),
	  tracepoint_id(tracepoint_id)
{
}

bpf_perf_event_handler::bpf_perf_event_handler(
	int cpu, int32_t sample_type, int64_t config,
	boost::interprocess::managed_shared_memory &mem)
	: type(bpf_event_type::PERF_TYPE_SOFTWARE),
	  _module_name(char_allocator(mem.get_segment_manager())),
	  sw_perf(boost::interprocess::make_managed_shared_ptr(
		  mem.construct<software_perf_event_data>(
			  boost::interprocess::anonymous_instance)(
			  cpu, config, sample_type, mem),
		  mem))

{
}

bool software_perf_event_data::has_data() const
{
	auto &ref = get_header_ref_const();
	return ref.data_tail != ref.data_head;
}

const perf_event_mmap_page &
software_perf_event_data::get_header_ref_const() const
{
	return *(perf_event_mmap_page *)(uintptr_t)(mmap_buffer.data());
}

int software_perf_event_data::output_data(const void *buf, size_t size)
{
	spdlog::debug("Handling perf event output data with size {}", size);
	auto &header = get_header_ref();
	perf_sample_raw head;
	head.header.type = PERF_RECORD_SAMPLE;
	head.header.size = sizeof(head) + size;
	head.header.misc = 0;
	head.size = size;
	int64_t data_head = smp_load_acquire_u64(&header.data_head);
	int64_t data_tail = header.data_tail;
	uint8_t *base_addr = (uint8_t *)mmap_buffer.data() + pagesize;
	int64_t available_size = mmap_size() - (data_head - data_tail);
	spdlog::debug("Data tail={}", data_tail);
	if (available_size == 0) {
		spdlog::debug("Upgraded available size to {}", mmap_size());
		available_size = mmap_size();
	}
	// If available_size is less or equal than head.header.size, just drop
	// the data. In this way, we'll never make data_head equals to
	// data_tail, at situation other than an empty buffer
	if (available_size <= head.header.size) {
		spdlog::debug(
			"Dropping data with size {}, available_size {}, required size {}",
			size, available_size, head.header.size);
		return 0;
	}
	auto &copy_size = head.header.size;
	if (copy_buffer.size() != copy_size)
		copy_buffer.resize(copy_size);

	memcpy(copy_buffer.data(), &head, sizeof(head));
	memcpy((uint8_t *)(copy_buffer.data()) + sizeof(head), buf, size);
	uint8_t *copy_start_1 = base_addr + (data_head & (mmap_size() - 1));
	if (copy_size + copy_start_1 <= base_addr + mmap_size()) {
		memcpy(copy_start_1, copy_buffer.data(), copy_size);
	} else {
		size_t len_first = base_addr + mmap_size() - copy_start_1;
		size_t len_second = copy_size - len_first;
		memcpy(copy_start_1, copy_buffer.data(), len_first);
		memcpy(base_addr, copy_buffer.data() + len_first, len_second);
	}
	uint64_t new_head = (data_head + copy_size);
	smp_store_release_u64(&header.data_head, new_head);
	spdlog::debug(
		"Data of size {}, total size {} outputed at head {}; new_head={} addr={:x}; available_size={}",
		size, copy_size, data_head, new_head,
		(uintptr_t)(base_addr + data_head), available_size);

	return 0;
}

perf_event_mmap_page &software_perf_event_data::get_header_ref()
{
	return *(perf_event_mmap_page *)(uintptr_t)(mmap_buffer.data());
}

software_perf_event_data::software_perf_event_data(
	int cpu, int64_t config, int32_t sample_type,
	boost::interprocess::managed_shared_memory &memory)
	: cpu(cpu), config(config), sample_type(sample_type),
	  pagesize(getpagesize()),
	  mmap_buffer(pagesize, memory.get_segment_manager()),
	  copy_buffer(pagesize, memory.get_segment_manager())
{
	perf_event_mmap_page &perf_header = get_header_ref();
	perf_header.data_offset = pagesize;
	perf_header.data_head = perf_header.data_tail = 0;
	perf_header.data_size = 0;
}

[[maybe_unused]] static inline int popcnt(uint64_t x)
{
	int ret = 0;
	while (x) {
		ret += (x & 1);
		x >>= 1;
	}
	return ret;
}

void *software_perf_event_data::ensure_mmap_buffer(size_t buffer_size)
{
	if (buffer_size > mmap_buffer.size()) {
		spdlog::debug("Expanding mmap buffer size to {}", buffer_size);
		mmap_buffer.resize(buffer_size);
		// Update data size in the mmap header
		get_header_ref().data_size = buffer_size - pagesize;
		assert(popcnt(buffer_size - pagesize) == 1 &&
		       "Data size of a perf event buffer must be power of 2");
	}
	return mmap_buffer.data();
}

size_t software_perf_event_data::mmap_size() const
{
	return mmap_buffer.size() - pagesize;
}

std::optional<software_perf_event_weak_ptr>
bpf_perf_event_handler::try_get_software_perf_data_weak_ptr() const
{
	if (sw_perf.has_value()) {
		return software_perf_event_weak_ptr(sw_perf.value());
	} else {
		return {};
	}
}

std::optional<void *>
bpf_perf_event_handler::try_get_software_perf_data_raw_buffer(
	size_t buffer_size) const
{
	if (sw_perf.has_value()) {
		return sw_perf.value()->ensure_mmap_buffer(buffer_size);
	} else {
		return {};
	}
}

} // namespace bpftime
