/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _RINGBUF_MAP_HPP
#define _RINGBUF_MAP_HPP
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/smart_ptr/deleter.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <pthread.h>
#include <vector>

namespace bpftime
{
class ringbuf {
	using vec_allocator = boost::interprocess::allocator<
		char,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using buf_vec = boost::interprocess::vector<char, vec_allocator>;
	using buf_vec_unique_ptr = boost::interprocess::managed_unique_ptr<
		buf_vec, boost::interprocess::managed_shared_memory>::type;
	uint32_t max_ent;
	uint32_t mask() const
	{
		return max_ent - 1;
	}
	boost::interprocess::offset_ptr<unsigned long> consumer_pos;
	boost::interprocess::offset_ptr<unsigned long> producer_pos;
	boost::interprocess::offset_ptr<uint8_t> data;
	// Start position of the synthetic tail-discard pad preceding the active
	// offset-0 record (or record_pos itself when offset 0 landed without
	// padding). No second offset-0 reservation can fit until the current
	// offset-0 record is committed and consumed.
	unsigned long wrap_pad_pos = 0;
#if defined(__linux__)
	// Kernel eventfd backing this ringbuf. The fd number is only valid in
	// the process that constructed the ringbuf (the syscall server); other
	// processes obtain a dup through the SCM_RIGHTS export socket.
	int kernel_eventfd = -1;
	int32_t eventfd_owner_pid = 0;
#endif
	// Process-shared reserve lock (PTHREAD_PROCESS_SHARED). On Linux
	// it is also robust: a killed holder hands the lock to the next
	// locker via EOWNERDEAD instead of wedging all producers.
	mutable pthread_mutex_t reserve_lock;
	// raw buffer
	buf_vec_unique_ptr raw_buffer;

    public:
	bool has_data() const;
	void *reserve(size_t size, int self_fd);
	void submit(const void *sample, bool discard);
	// libbpf-adaptive wakeup decision: BPF_RB_FORCE_WAKEUP always wakes,
	// BPF_RB_NO_WAKEUP never wakes, otherwise wake only when the consumer
	// had drained every record before this one (it is likely blocked in
	// epoll_wait). `sample` is the pointer returned by reserve().
	bool should_wakeup(const void *sample, uint64_t flags) const;
	int fetch_data(int (*)(void *ctx, void *data, size_t len), void *ctx);
#if defined(__linux__)
	int get_kernel_eventfd() const
	{
		return kernel_eventfd;
	}
	int32_t get_eventfd_owner_pid() const
	{
		return eventfd_owner_pid;
	}
#endif
	ringbuf(uint32_t max_ent,
		boost::interprocess::managed_shared_memory &memory);
	friend class ringbuf_map_impl;
	~ringbuf();

	// Test-only seam: hold the lock across a fork+SIGKILL so robust
	// mutex recovery is exercised deterministically.
	void lock_for_testing() const
	{
		pthread_mutex_lock(&reserve_lock);
	}
	void unlock_for_testing() const
	{
		pthread_mutex_unlock(&reserve_lock);
	}
};

using ringbuf_shared_ptr = boost::interprocess::managed_shared_ptr<
	ringbuf,
	boost::interprocess::managed_shared_memory::segment_manager>::type;
using ringbuf_weak_ptr = boost::interprocess::managed_weak_ptr<
	ringbuf,
	boost::interprocess::managed_shared_memory::segment_manager>::type;

// implementation of ringbuf map
class ringbuf_map_impl {
	ringbuf_shared_ptr ringbuf_impl;

    public:
	const static bool should_lock = false;
	ringbuf_map_impl(uint32_t max_ent,
			 boost::interprocess::managed_shared_memory &memory);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
	ringbuf_weak_ptr create_impl_weak_ptr();
	ringbuf_shared_ptr create_impl_shared_ptr();

	void *get_consumer_page() const;
	void *get_producer_page() const;
	void *reserve(size_t size, int self_fd);
	void submit(const void *sample, bool discard);
#if defined(__linux__)
	int get_kernel_eventfd() const
	{
		return ringbuf_impl->get_kernel_eventfd();
	}
	int32_t get_eventfd_owner_pid() const
	{
		return ringbuf_impl->get_eventfd_owner_pid();
	}
	bool should_wakeup(const void *sample, uint64_t flags) const
	{
		return ringbuf_impl->should_wakeup(sample, flags);
	}
#endif
};

} // namespace bpftime
#endif
