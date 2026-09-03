#include "nv_gpu_ringbuf_map.hpp"
#include "bpftime_shm_internal.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
using namespace bpftime;

namespace
{
uint64_t checked_add(uint64_t lhs, uint64_t rhs)
{
	if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
		throw std::overflow_error("GPU ring buffer size overflow");
	return lhs + rhs;
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs)
{
	if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
		throw std::overflow_error("GPU ring buffer size overflow");
	return lhs * rhs;
}

uint64_t align_u64(uint64_t value)
{
	const uint64_t mask = alignof(uint64_t) - 1;
	return checked_add(value, mask) & ~mask;
}
} // namespace

nv_gpu_ringbuf_map_impl::nv_gpu_ringbuf_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint64_t value_size,
	uint64_t max_entries, uint64_t thread_count)
	: data_buffer(memory.get_segment_manager()),
	  agent_gpu_shared_mem(memory.get_segment_manager()),
	  value_size(value_size), max_entries(max_entries),
	  thread_count(thread_count), local_buffer(memory.get_segment_manager())
{
	record_stride = align_u64(checked_add(value_size, sizeof(uint64_t)));
	entry_size = checked_add(sizeof(ringbuf_header),
				 checked_mul(record_stride, max_entries));
	const uint64_t data_size = checked_add(
		checked_mul(entry_size, thread_count),
		sizeof(ringbuf_error_counters));
	if (entry_size > std::numeric_limits<size_t>::max() ||
	    data_size > std::numeric_limits<size_t>::max())
		throw std::length_error("GPU ring buffer exceeds host size_t");
	local_buffer.resize(entry_size);
	data_buffer.resize(data_size);
}

CUdeviceptr
nv_gpu_ringbuf_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	if (shm_holder.global_shared_memory.get_open_type() !=
	    shm_open_type::SHM_REMOVE_AND_CREATE) {
		int pid = getpid();
		if (auto itr = agent_gpu_shared_mem.find(pid);
		    itr == agent_gpu_shared_mem.end()) {
			SPDLOG_INFO(
				"Initializing nv_gpu_ringbuf_map_impl at pid {}",
				pid);
			CUdeviceptr device_ptr = 0;
			if (auto err = cuMemHostGetDevicePointer(
				    &device_ptr, (void *)data_buffer.data(), 0);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to map host ringbuf buffer into device address space, error={}",
					(int)err);
				throw std::runtime_error(
					"Unable to map host ringbuf buffer into device address space");
			}
			SPDLOG_INFO("Mapped GPU memory for gpu ringbuf map: {}",
				    (uintptr_t)device_ptr);
			agent_gpu_shared_mem[pid] = device_ptr;
		}
		return agent_gpu_shared_mem[pid];
	} else {
		return (CUdeviceptr)data_buffer.data();
	}
}

int nv_gpu_ringbuf_map_impl::drain_data(
	const std::function<void(const void *, uint64_t)> &fn)
{
	if (!fn || max_entries == 0)
		return -EINVAL;
	std::atomic_thread_fence(std::memory_order_acquire);
	uint64_t drained = 0;
	for (uint64_t i = 0; i < thread_count; i++) {
		auto header = (ringbuf_header *)(uintptr_t)(data_buffer.data() +
							    i * entry_size);
		if (__atomic_load_n(&header->dirty, __ATOMIC_ACQUIRE))
			continue;

		uint64_t head = __atomic_load_n(&header->head, __ATOMIC_ACQUIRE);
		const uint64_t tail =
			__atomic_load_n(&header->tail, __ATOMIC_ACQUIRE);
		if (tail < head || tail - head > max_entries)
			return -EOVERFLOW;
		while (head < tail) {
			const auto real_head = head % max_entries;
			auto buffer_start =
				((char *)header) + sizeof(ringbuf_header) +
				real_head * record_stride;
			const uint64_t size =
				*(uint64_t *)(uintptr_t)buffer_start;
			if (size > value_size)
				return -EMSGSIZE;
			fn(buffer_start + sizeof(uint64_t), size);
			head++;
			__atomic_store_n(&header->head, head, __ATOMIC_RELEASE);
			drained++;
		}
	}

	return drained > INT_MAX ? -EOVERFLOW : (int)drained;
}

int nv_gpu_ringbuf_map_impl::get_stats(bpftime_gpu_ringbuf_stats *stats) const
{
	if (!stats)
		return -EINVAL;

	*stats = {};
	stats->value_size = value_size;
	stats->entries_per_thread = max_entries;
	stats->allocated_thread_slots = thread_count;
	for (uint64_t i = 0; i < thread_count; i++) {
		const auto *header = (const ringbuf_header *)(uintptr_t)(
			data_buffer.data() + i * entry_size);
		const uint64_t head =
			__atomic_load_n(&header->head, __ATOMIC_ACQUIRE);
		const uint64_t tail =
			__atomic_load_n(&header->tail, __ATOMIC_ACQUIRE);
		stats->collected_records += head;
		stats->committed_records += tail;
		if (tail >= head)
			stats->pending_records += tail - head;
		else
			stats->other_drops++;
		stats->dirty_slots +=
			__atomic_load_n(&header->dirty, __ATOMIC_ACQUIRE) != 0;
	}

	const auto *errors = (const ringbuf_error_counters *)(uintptr_t)(
		data_buffer.data() + entry_size * thread_count);
	stats->oob_drops =
		__atomic_load_n(&errors->oob_drops, __ATOMIC_ACQUIRE);
	stats->full_drops =
		__atomic_load_n(&errors->full_drops, __ATOMIC_ACQUIRE);
	stats->bad_size_drops =
		__atomic_load_n(&errors->bad_size_drops, __ATOMIC_ACQUIRE);
	stats->other_drops +=
		__atomic_load_n(&errors->other_drops, __ATOMIC_ACQUIRE);
	return 0;
}

nv_gpu_ringbuf_map_impl::~nv_gpu_ringbuf_map_impl()
{
}

void *nv_gpu_ringbuf_map_impl::elem_lookup(const void *key)
{
	SPDLOG_ERROR("Element lookup is not supported by gpu ringbuf map");
	errno = -ENOTSUP;
	return nullptr;
}

long nv_gpu_ringbuf_map_impl::elem_update(const void *key, const void *value,
					  uint64_t flags)
{
	SPDLOG_ERROR("Element update is not supported by gpu ringbuf map");
	errno = -ENOTSUP;
	return -1;
}

long nv_gpu_ringbuf_map_impl::elem_delete(const void *key)
{
	SPDLOG_ERROR("Element delete is not supported by gpu ringbuf map");
	errno = -ENOTSUP;
	return -1;
}

int nv_gpu_ringbuf_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_ERROR("Get next key is not supported by gpu ringbuf map");
	errno = -ENOTSUP;
	return -1;
}
