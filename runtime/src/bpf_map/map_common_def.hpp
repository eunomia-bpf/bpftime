#ifndef _MAP_COMMON_DEF_HPP
#define _MAP_COMMON_DEF_HPP
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
namespace bpftime
{

using bytes_vec_allocator = boost::interprocess::allocator<
	uint8_t, boost::interprocess::managed_shared_memory::segment_manager>;
using bytes_vec = boost::interprocess::vector<uint8_t, bytes_vec_allocator>;

} // namespace bpftime

#endif
