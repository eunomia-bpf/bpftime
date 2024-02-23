/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _HANDLER_MANAGER_HPP
#define _HANDLER_MANAGER_HPP
#include "handlers/epoll_handler.hpp"
#include "spdlog/spdlog.h"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sched.h>
#include <variant>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/containers/set.hpp>
#include <handlers/link_handler.hpp>
#include <handlers/prog_handler.hpp>
#include <handlers/map_handler.hpp>
#include <handlers/perf_event_handler.hpp>

#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#endif
namespace bpftime
{
namespace shm_common
{
using managed_shared_memory = boost::interprocess::managed_shared_memory;
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

const size_t DEFAULT_MAX_ID = 1024 * 6;

struct unused_handler {};

using handler_variant =
	std::variant<unused_handler, bpf_map_handler, bpf_link_handler,
		     bpf_prog_handler, bpf_perf_event_handler, epoll_handler>;

using handler_variant_allocator =
	boost::interprocess::allocator<handler_variant, managed_shared_memory::segment_manager>;

using handler_variant_vector =
	boost::interprocess::vector<handler_variant, handler_variant_allocator>;

// handler manager for keep bpf maps and progs fds
// This struct will be put on shared memory
class handler_manager {
    public:
	handler_manager(managed_shared_memory &mem,
			size_t max_id_count = DEFAULT_MAX_ID);

	~handler_manager();
	// Get a reference to the handler of the certain id
	const handler_variant &get_handler(int id) const;

	const handler_variant &operator[](int idx) const;
	// Get the max handler size
	std::size_t size() const;
	// Set the handler at the certain id. Destroying the previous one
	int set_handler(int id, handler_variant &&handler,
			managed_shared_memory &memory);
	// Check if handler with the specified id is allocated
	bool is_allocated(int id) const;

	// Try to find a minimal unused index for handler
	int find_minimal_unused_idx() const;

	// Destroy the handler at the certain id
	void destroy_handler_at(int id, managed_shared_memory &memory);

	// Destroy all handlers held by this handler_manager
	void clear_all(managed_shared_memory &memory);

	handler_manager(const handler_manager &) = delete;
	handler_manager(handler_manager &&) noexcept = default;
	handler_manager &operator=(const handler_manager &) = delete;
	handler_manager &operator=(handler_manager &&) noexcept = default;

    private:
	handler_variant_vector handlers;
};

} // namespace shm_common
} // namespace bpftime

#endif
