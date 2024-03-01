#ifndef _BPFTIME_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_ATTACH_PRIVATE_DATA_HPP

#include <string_view>
#include "spdlog/spdlog.h"
#include <stdexcept>
namespace bpftime
{
namespace attach
{
// A base class for all attach-independent private data
struct attach_private_data {
	virtual ~attach_private_data(){};
	// Initialize this private data structure from a string.
	// This function should be implemented by things like
	// `uprobe_attach_private_data` or `syscall_attach_private_data` or
	// other ones. In this way, we provide a unified interface to create
	// private data for different attaches.
	virtual int initialize_from_string(const std::string_view &sv)
	{
		SPDLOG_ERROR(
			"Not implemented: attach_private_data::initialize_from_string");
		throw std::runtime_error(
			"attach_private_data::initialize_from_string");
	}
};
} // namespace attach
} // namespace bpftime
#endif
