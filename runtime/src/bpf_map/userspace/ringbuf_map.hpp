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
#include <functional>
#include <optional>
#include <vector>

namespace bpftime
{
using sharable_mutex_ptr = boost::interprocess::managed_unique_ptr<
	boost::interprocess::interprocess_sharable_mutex,
	boost::interprocess::managed_shared_memory>::type;

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
	// Guard for reserving memory
	mutable sharable_mutex_ptr reserve_mutex;
	// raw buffer
	buf_vec_unique_ptr raw_buffer;

    public:
	bool has_data() const;
	void *reserve(size_t size, int self_fd);
	void submit(const void *sample, bool discard);
	int fetch_data(std::function<int(void *, int)>);
	ringbuf(uint32_t max_ent,
		boost::interprocess::managed_shared_memory &memory);
	friend class ringbuf_map_impl;
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
};

} // namespace bpftime
#endif
