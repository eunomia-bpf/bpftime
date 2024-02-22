/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _EPOLL_HANDLER_HPP
#define _EPOLL_HANDLER_HPP
#include "handlers/perf_event_handler.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <maps/userspace/ringbuf_map.hpp>
#include <sys/epoll.h>
#include <variant>
namespace bpftime
{
namespace shm_common
{
// `File` types that epoll could hold
using file_ptr_variant =
	std::variant<software_perf_event_weak_ptr, ringbuf_weak_ptr>;
// `Fd` ptr and the related user-defined data
struct epoll_file {
	file_ptr_variant file;
	epoll_data_t data;
	epoll_file(const file_ptr_variant &&ptr);
	epoll_file(const file_ptr_variant &&ptr, epoll_data_t data);
};
using file_allocator = boost::interprocess::allocator<
	epoll_file, boost::interprocess::managed_shared_memory::segment_manager>;
using file_vector = boost::interprocess::vector<epoll_file, file_allocator>;
// Represent an epoll instance in the shared memory
struct epoll_handler {
    public:
	mutable file_vector files;
	epoll_handler(boost::interprocess::managed_shared_memory &memory);
};
} // namespace shm_common
} // namespace bpftime
#endif
