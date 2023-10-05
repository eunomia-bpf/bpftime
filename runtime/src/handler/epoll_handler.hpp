#ifndef _EPOLL_HANDLER_HPP
#define _EPOLL_HANDLER_HPP
#include <boost/interprocess/containers/vector.hpp>
#include <bpf_map/ringbuf_map.hpp>
namespace bpftime
{
using vec_allocator = boost::interprocess::allocator<
	ringbuf_weak_ptr,
	boost::interprocess::managed_shared_memory::segment_manager>;
using ringbuf_vec =
	boost::interprocess::vector<ringbuf_weak_ptr, vec_allocator>;
struct epoll_handler {
    public:
	ringbuf_vec holding_ringbufs;
	epoll_handler(boost::interprocess::managed_shared_memory &memory);
};
} // namespace bpftime
#endif
