/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _HANDLER_MANAGER_HPP
#define _HANDLER_MANAGER_HPP
#include "bpftime_config.hpp"
#include "handler/epoll_handler.hpp"
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
#include <bpftime_shm.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/containers/set.hpp>
#include <handler/link_handler.hpp>
#include <handler/prog_handler.hpp>
#include <handler/map_handler.hpp>
#include <handler/perf_event_handler.hpp>

#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#endif
namespace bpftime
{
using managed_shared_memory = boost::interprocess::managed_shared_memory;
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

const size_t DEFAULT_MAX_FD = 1024 * 6;

struct unused_handler {};

using boost::interprocess::allocator;
using boost::interprocess::vector;

constexpr const char *DEFAULT_GLOBAL_SHM_NAME = "bpftime_maps_shm";
constexpr const char *DEFAULT_GLOBAL_HANDLER_NAME = "bpftime_handler";
constexpr const char *DEFAULT_SYSCALL_PID_SET_NAME = "bpftime_syscall_pid_set";
constexpr const char *DEFAULT_AGENT_CONFIG_NAME = "bpftime_agent_config";
constexpr const char* DEFAULT_ALIVE_AGENT_PIDS_NAME = "bpftime_alive_agent_pids";
inline const char *get_global_shm_name()
{
	const char *name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	if (name == nullptr) {
		return DEFAULT_GLOBAL_SHM_NAME;
	}
	return name;
}

struct shm_remove {
	shm_remove()
	{
		boost::interprocess::shared_memory_object::remove(
			get_global_shm_name());
	}
	shm_remove(const char *name)
	{
		boost::interprocess::shared_memory_object::remove(name);
	}
	~shm_remove()
	{
		SPDLOG_INFO("Destroy shm {}", get_global_shm_name());
		boost::interprocess::shared_memory_object::remove(
			get_global_shm_name());
	}
};

using handler_variant =
	std::variant<unused_handler, bpf_map_handler, bpf_link_handler,
		     bpf_prog_handler, bpf_perf_event_handler, epoll_handler>;

using handler_variant_allocator =
	allocator<handler_variant, managed_shared_memory::segment_manager>;

using handler_variant_vector =
	boost::interprocess::vector<handler_variant, handler_variant_allocator>;

// handler manager for keep bpf maps and progs fds
// This struct will be put on shared memory
class handler_manager {
    public:
	handler_manager(managed_shared_memory &mem,
			size_t max_fd_count = DEFAULT_MAX_FD);

	~handler_manager();

	const handler_variant &get_handler(int fd) const;

	const handler_variant &operator[](int idx) const;
	std::size_t size() const;

	int set_handler_at_empty_slot(handler_variant &&handler,
				      managed_shared_memory &memory);

	int set_handler(int fd, handler_variant &&handler,
			managed_shared_memory &memory);

	bool is_allocated(int fd) const;

	int find_minimal_unused_idx() const;

	void clear_id_at(int fd, managed_shared_memory &memory);

	void clear_all(managed_shared_memory &memory);

	handler_manager(const handler_manager &) = delete;
	handler_manager(handler_manager &&) noexcept = default;
	handler_manager &operator=(const handler_manager &) = delete;
	handler_manager &operator=(handler_manager &&) noexcept = default;

    private:
	handler_variant_vector handlers;
};

} // namespace bpftime

#endif
