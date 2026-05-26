/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#if __linux__
#include "linux/perf_event.h"
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include "spdlog/spdlog.h"
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <cstring>
#include <handler/perf_event_handler.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <unistd.h>
#include <spdlog/fmt/bin_to_hex.h>
#include <unordered_map>
#include <variant>
#if __linux__
#include <sys/syscall.h>
#elif __APPLE__
#include <pthread.h>
#endif

#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#define READ_ONCE_U64(x) (*(volatile uint64_t *)&x)
#define WRITE_ONCE_U64(x, v) (*(volatile uint64_t *)&x) = (v)

#if defined(__x86_64__)
#define barrier() asm volatile("" ::: "memory")
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
#elif defined(__aarch64__)
// https://github.com/torvalds/linux/blob/master/tools/arch/arm64/include/asm/barrier.h
#define smp_store_release_u64(p, v)                                            \
	do {                                                                   \
		asm volatile("stlr %1, %0" : "=Q"(*p) : "r"(v) : "memory");    \
	} while (0)
#define smp_load_acquire_u64(p)                                                \
	({                                                                     \
		uint64_t ___p;                                                 \
		asm volatile("ldar %0, %1" : "=r"(___p) : "Q"(*p) : "memory"); \
		___p;                                                          \
	})
#else
#error Only supports x86_64 and aarch64
#endif

namespace bpftime
{
namespace
{
struct spin_lock_guard {
	pthread_spinlock_t &lock;
	explicit spin_lock_guard(pthread_spinlock_t &lock) : lock(lock)
	{
		pthread_spin_lock(&lock);
	}
	~spin_lock_guard()
	{
		pthread_spin_unlock(&lock);
	}
	spin_lock_guard(const spin_lock_guard &) = delete;
	spin_lock_guard &operator=(const spin_lock_guard &) = delete;
};

int64_t current_thread_id()
{
#if __linux__
	return (int64_t)syscall(SYS_gettid);
#elif __APPLE__
	uint64_t tid = 0;
	pthread_threadid_np(nullptr, &tid);
	return (int64_t)tid;
#else
	return 0;
#endif
}

thread_local std::unordered_map<const software_perf_event_data *,
				software_perf_event_shard *>
	software_perf_event_shard_cache;
} // namespace

bpf_perf_event_handler::bpf_perf_event_handler(
	int type, const char *attach_arg,
	boost::interprocess::managed_shared_memory &mem)
	: type(type),
	  data(custom_perf_event_data{
		  .attach_argument = boost_shm_string(
			  char_allocator(mem.get_segment_manager())) })
{
	std::get<custom_perf_event_data>(data).attach_argument = attach_arg;
}

// attach to replace or filter self define types
bpf_perf_event_handler::bpf_perf_event_handler(
	bpf_event_type type, uint64_t offset, int pid, const char *module_name,
	boost::interprocess::managed_shared_memory &mem, bool default_enabled)
	: type((int)type), enabled(default_enabled),
	  data(uprobe_perf_event_data{
		  .offset = offset,
		  .pid = pid,
		  ._module_name = boost_shm_string(
			  char_allocator(mem.get_segment_manager())) })
{
	std::get<uprobe_perf_event_data>(data)._module_name = module_name;
}

// create uprobe/uretprobe with new perf event attr
bpf_perf_event_handler::bpf_perf_event_handler(
	bool is_retprobe, uint64_t offset, int pid, const char *module_name,
	size_t ref_ctr_off, boost::interprocess::managed_shared_memory &mem)
	: data(uprobe_perf_event_data{
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
	std::get<uprobe_perf_event_data>(data)._module_name = module_name;
	SPDLOG_INFO(
		"Created uprobe/uretprobe perf event handler, module name {}, offset {:x}",
		module_name, offset);
}

bpf_perf_event_handler::bpf_perf_event_handler(
	bool is_retprobe, uint64_t addr, const char *func_name,
	size_t ref_ctr_off, boost::interprocess::managed_shared_memory &mem)
	: data(kprobe_perf_event_data{
		  .func_name = boost_shm_string(
			  char_allocator(mem.get_segment_manager())),
		  .addr = addr,
		  .ref_ctr_off = ref_ctr_off })
{
	std::get<kprobe_perf_event_data>(data).func_name = func_name;
	if (is_retprobe) {
		type = (int)bpf_event_type::BPF_TYPE_KRETPROBE;
	} else {
		type = (int)bpf_event_type::BPF_TYPE_KPROBE;
	}
	SPDLOG_INFO(
		"Created kprobe/kretprobe perf event handler, func_name {}, addr {:x}",
		func_name, addr);
}
// create tracepoint
bpf_perf_event_handler::bpf_perf_event_handler(
	int pid, int32_t tracepoint_id,
	boost::interprocess::managed_shared_memory &mem)
	: type((int)bpf_event_type::PERF_TYPE_TRACEPOINT),
	  data(tracepoint_perf_event_data{ .pid = pid,
					   .tracepoint_id = tracepoint_id })
{
}

bpf_perf_event_handler::bpf_perf_event_handler(
	int cpu, int32_t sample_type, int64_t config,
	boost::interprocess::managed_shared_memory &mem)
	: type((int)bpf_event_type::PERF_TYPE_SOFTWARE),
	  data(boost::interprocess::make_managed_shared_ptr(
		  mem.construct<software_perf_event_data>(
			  boost::interprocess::anonymous_instance)(
			  cpu, config, sample_type, mem),
		  mem))

{
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

software_perf_event_buffer::software_perf_event_buffer(int pagesize,
						       segment_manager *manager)
	: pagesize(pagesize), mmap_buffer(pagesize, manager),
	  copy_buffer(pagesize, manager)
{
	perf_event_mmap_page &perf_header = get_header_ref();
	perf_header.data_offset = pagesize;
	perf_header.data_head = perf_header.data_tail = 0;
	perf_header.data_size = 0;
}

const perf_event_mmap_page &
software_perf_event_buffer::get_header_ref_const() const
{
	return *(perf_event_mmap_page *)(uintptr_t)(mmap_buffer.data());
}

perf_event_mmap_page &software_perf_event_buffer::get_header_ref()
{
	return *(perf_event_mmap_page *)(uintptr_t)(mmap_buffer.data());
}

bool software_perf_event_buffer::has_data() const
{
	auto &ref = get_header_ref_const();
	return smp_load_acquire_u64(&ref.data_tail) !=
	       smp_load_acquire_u64(&ref.data_head);
}

void *software_perf_event_buffer::ensure_mmap_buffer(size_t buffer_size)
{
	if (buffer_size > mmap_buffer.size()) {
		SPDLOG_DEBUG("Expanding mmap buffer size to {}", buffer_size);
		mmap_buffer.resize(buffer_size);
		// Update data size in the mmap header
		get_header_ref().data_size = buffer_size - pagesize;
		if (popcnt(buffer_size - pagesize) != 1) {
			SPDLOG_ERROR(
				"Data size of a perf event buffer must be power of 2");
			return nullptr;
		}
	}
	return mmap_buffer.data();
}

size_t software_perf_event_buffer::mmap_size() const
{
	return mmap_buffer.size() - pagesize;
}

bool software_perf_event_buffer::append_record(const void *record,
					       size_t record_size)
{
	if (mmap_size() == 0 || record_size > mmap_size()) {
		return false;
	}
	auto &header = get_header_ref();
	uint64_t data_head = smp_load_acquire_u64(&header.data_head);
	uint64_t data_tail = smp_load_acquire_u64(&header.data_tail);
	uint8_t *base_addr = (uint8_t *)mmap_buffer.data() + pagesize;
	int64_t available_size =
		(int64_t)mmap_size() - (int64_t)(data_head - data_tail);
	if (available_size == 0) {
		available_size = mmap_size();
	}
	// Keep one byte empty so head == tail only represents an empty buffer.
	if (available_size <= (int64_t)record_size) {
		SPDLOG_DEBUG(
			"Dropping perf record with size {}, available_size {}",
			record_size, available_size);
		return false;
	}

	uint8_t *copy_start_1 = base_addr + (data_head & (mmap_size() - 1));
	if (record_size + copy_start_1 <= base_addr + mmap_size()) {
		memcpy(copy_start_1, record, record_size);
	} else {
		size_t len_first = base_addr + mmap_size() - copy_start_1;
		size_t len_second = record_size - len_first;
		memcpy(copy_start_1, record, len_first);
		memcpy(base_addr, (const uint8_t *)record + len_first,
		       len_second);
	}
	uint64_t new_head = data_head + record_size;
	smp_store_release_u64(&header.data_head, new_head);
	SPDLOG_DEBUG(
		"Perf record of size {} outputed at head {}; new_head={} addr={:x}; available_size={}",
		record_size, data_head, new_head,
		(uintptr_t)(base_addr + data_head), available_size);
	return true;
}

int software_perf_event_buffer::output_data(const void *buf, size_t size)
{
	SPDLOG_DEBUG("Handling perf event output data with size {}", size);
	perf_sample_raw head;
	head.header.type = PERF_RECORD_SAMPLE;
	head.header.size = sizeof(head) + size;
	head.header.misc = 0;
	head.size = size;

	auto copy_size = head.header.size;
	if (copy_buffer.size() != copy_size)
		copy_buffer.resize(copy_size);
	memcpy(copy_buffer.data(), &head, sizeof(head));
	memcpy((uint8_t *)(copy_buffer.data()) + sizeof(head), buf, size);
	append_record(copy_buffer.data(), copy_size);
	return 0;
}

void software_perf_event_buffer::copy_from_ring(uint64_t offset, void *dst,
						size_t size) const
{
	uint8_t *base_addr = (uint8_t *)mmap_buffer.data() + pagesize;
	uint8_t *copy_start_1 = base_addr + (offset & (mmap_size() - 1));
	if (size + copy_start_1 <= base_addr + mmap_size()) {
		memcpy(dst, copy_start_1, size);
	} else {
		size_t len_first = base_addr + mmap_size() - copy_start_1;
		size_t len_second = size - len_first;
		memcpy(dst, copy_start_1, len_first);
		memcpy((uint8_t *)dst + len_first, base_addr, len_second);
	}
}

bool software_perf_event_buffer::copy_next_record_to(
	software_perf_event_buffer &dst)
{
	if (mmap_size() == 0) {
		return false;
	}
	auto &header = get_header_ref();
	uint64_t data_tail = smp_load_acquire_u64(&header.data_tail);
	uint64_t data_head = smp_load_acquire_u64(&header.data_head);
	if (data_tail == data_head) {
		return false;
	}
	if (data_head < data_tail) {
		SPDLOG_ERROR("Invalid perf buffer state: head {} < tail {}",
			     data_head, data_tail);
		smp_store_release_u64(&header.data_tail, data_head);
		return false;
	}

	perf_event_header record_header;
	copy_from_ring(data_tail, &record_header, sizeof(record_header));
	if (record_header.size < sizeof(record_header) ||
	    record_header.size > mmap_size()) {
		SPDLOG_ERROR("Invalid perf record size {}, dropping shard data",
			     record_header.size);
		smp_store_release_u64(&header.data_tail, data_head);
		return false;
	}
	if (record_header.size > data_head - data_tail) {
		return false;
	}

	if (copy_buffer.size() != record_header.size)
		copy_buffer.resize(record_header.size);
	copy_from_ring(data_tail, copy_buffer.data(), record_header.size);
	if (!dst.append_record(copy_buffer.data(), record_header.size)) {
		return false;
	}
	smp_store_release_u64(&header.data_tail,
			      data_tail + record_header.size);
	return true;
}

software_perf_event_shard::software_perf_event_shard(
	int pid, int64_t tid, uint64_t generation, int pagesize,
	software_perf_event_buffer::segment_manager *manager,
	size_t buffer_size)
	: pid(pid), tid(tid), generation(generation), buffer(pagesize, manager)
{
	buffer.ensure_mmap_buffer(buffer_size);
}

software_perf_event_data::software_perf_event_data(
	int cpu, int64_t config, int32_t sample_type,
	boost::interprocess::managed_shared_memory &memory)
	: cpu(cpu), config(config), sample_type(sample_type),
	  pagesize(getpagesize()),
	  consumer_buffer(pagesize, memory.get_segment_manager()),
	  producer_shards(memory.get_segment_manager())
{
	pthread_spin_init(&shard_lock, PTHREAD_PROCESS_SHARED);
}

software_perf_event_data::~software_perf_event_data()
{
	pthread_spin_destroy(&shard_lock);
}

software_perf_event_shard &software_perf_event_data::get_current_thread_shard()
{
	const int pid = getpid();
	const int64_t tid = current_thread_id();
	if (auto itr = software_perf_event_shard_cache.find(this);
	    itr != software_perf_event_shard_cache.end() &&
	    itr->second->pid == pid && itr->second->tid == tid) {
		return *itr->second;
	}

	spin_lock_guard guard(shard_lock);
	for (auto &shard : producer_shards) {
		if (shard.pid == pid && shard.tid == tid) {
			software_perf_event_shard_cache[this] = &shard;
			return shard;
		}
	}

	auto *manager = producer_shards.get_allocator().get_segment_manager();
	producer_shards.emplace_back(pid, tid, next_generation++, pagesize,
				     manager,
				     consumer_buffer.mmap_buffer.size());
	auto &shard = producer_shards.back();
	software_perf_event_shard_cache[this] = &shard;
	return shard;
}

void software_perf_event_data::drain_producer_shards()
{
	spin_lock_guard guard(shard_lock);
	for (auto &shard : producer_shards) {
		while (shard.buffer.copy_next_record_to(consumer_buffer)) {
		}
	}
}

bool software_perf_event_data::has_data()
{
	drain_producer_shards();
	return consumer_buffer.has_data();
}

const perf_event_mmap_page &
software_perf_event_data::get_header_ref_const() const
{
	return consumer_buffer.get_header_ref_const();
}

int software_perf_event_data::output_data(const void *buf, size_t size)
{
	return get_current_thread_shard().buffer.output_data(buf, size);
}

perf_event_mmap_page &software_perf_event_data::get_header_ref()
{
	return consumer_buffer.get_header_ref();
}

void *software_perf_event_data::ensure_mmap_buffer(size_t buffer_size)
{
	void *result = consumer_buffer.ensure_mmap_buffer(buffer_size);
	if (result == nullptr) {
		return nullptr;
	}
	spin_lock_guard guard(shard_lock);
	for (auto &shard : producer_shards) {
		if (shard.buffer.ensure_mmap_buffer(buffer_size) == nullptr) {
			return nullptr;
		}
	}
	return result;
}

size_t software_perf_event_data::mmap_size() const
{
	return consumer_buffer.mmap_size();
}

std::optional<software_perf_event_weak_ptr>
bpf_perf_event_handler::try_get_software_perf_data_weak_ptr() const
{
	if (std::holds_alternative<software_perf_event_shared_ptr>(data)) {
		return software_perf_event_weak_ptr(
			std::get<software_perf_event_shared_ptr>(data));
	} else {
		return {};
	}
}

std::optional<void *>
bpf_perf_event_handler::try_get_software_perf_data_raw_buffer(
	size_t buffer_size) const
{
	if (std::holds_alternative<software_perf_event_shared_ptr>(data)) {
		return std::get<software_perf_event_shared_ptr>(data)
			->ensure_mmap_buffer(buffer_size);
	} else {
		return {};
	}
}

} // namespace bpftime
