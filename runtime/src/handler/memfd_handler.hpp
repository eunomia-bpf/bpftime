#ifndef _BPFTIME_MEMFD_HANDLER_HPP
#define _BPFTIME_MEMFD_HANDLER_HPP

#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
namespace bpftime
{

using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

struct memfd_handler {
	int flags;
	boost_shm_string name;
	memfd_handler(const char *name, int flags,
		      boost::interprocess::managed_shared_memory &memory);
};
} // namespace bpftime

#endif
