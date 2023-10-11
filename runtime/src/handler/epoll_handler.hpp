#ifndef _EPOLL_HANDLER_HPP
#define _EPOLL_HANDLER_HPP
#include "handler/perf_event_handler.hpp"
#include <boost/interprocess/containers/vector.hpp>
#include <bpf_map/ringbuf_map.hpp>
namespace bpftime
{
using ringbuf_vec_allocator = boost::interprocess::allocator<
	ringbuf_weak_ptr,
	boost::interprocess::managed_shared_memory::segment_manager>;
using ringbuf_vec =
	boost::interprocess::vector<ringbuf_weak_ptr, ringbuf_vec_allocator>;

using sw_data_vec_allocator = boost::interprocess::allocator<
	software_perf_event_weak_ptr,
	boost::interprocess::managed_shared_memory::segment_manager>;
using sw_data_vec = boost::interprocess::vector<software_perf_event_weak_ptr,
						sw_data_vec_allocator>;

struct epoll_handler {
    public:
	mutable ringbuf_vec holding_ringbufs;
	mutable sw_data_vec holding_software_perf_events;
	epoll_handler(boost::interprocess::managed_shared_memory &memory);
};
} // namespace bpftime
#endif
