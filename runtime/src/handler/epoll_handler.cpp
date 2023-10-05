#include <handler/epoll_handler.hpp>
namespace bpftime
{
epoll_handler::epoll_handler(boost::interprocess::managed_shared_memory &memory)
	: holding_ringbufs(memory.get_segment_manager())
{
}
} // namespace bpftime
