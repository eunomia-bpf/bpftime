/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <handler/epoll_handler.hpp>
#if __linux__
#include <sys/epoll.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
namespace bpftime
{
epoll_handler::epoll_handler(boost::interprocess::managed_shared_memory &memory)
	: files(memory.get_segment_manager())

{
	epoll_event ev;
}

epoll_file::epoll_file(const file_ptr_variant &&ptr, int origin_fd)
	: file(std::move(ptr)), origin_fd(origin_fd)
{
}
epoll_file::epoll_file(const file_ptr_variant &&ptr, epoll_data_t data,
		       int origin_fd)
	: file(std::move(ptr)), data(data), origin_fd(origin_fd)
{
}

} // namespace bpftime
