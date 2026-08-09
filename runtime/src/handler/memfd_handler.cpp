#include "memfd_handler.hpp"

using namespace bpftime;

memfd_handler::memfd_handler(const char *name, int flags,
			     boost::interprocess::managed_shared_memory &memory)
	: name(char_allocator(memory.get_segment_manager()))
{
	this->name = name;
	this->flags = flags;
}
