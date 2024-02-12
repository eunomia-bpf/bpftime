#ifndef _BPFTIME_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_ATTACH_PRIVATE_DATA_HPP

#include <string_view>
namespace bpftime
{
namespace attach
{
// A base class for all attach-independent private data
struct attach_private_data {
	virtual ~attach_private_data();
	// Initialize this private data structure from a string.
	// This function should be implemented by things like
	// `uprobe_attach_private_data` or `syscall_attach_private_data` or
	// other ones So we provide a unified interface to create private data
	// for different attaches
	virtual int initialize_from_string(const std::string_view &sv);
};
} // namespace attach
} // namespace bpftime
#endif
