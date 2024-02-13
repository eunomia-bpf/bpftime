#ifndef _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#include <attach_private_data.hpp>
namespace bpftime
{
namespace attach
{
// Private data of syscall trace attach
struct syscall_trace_attach_private_data : public attach_private_data {
	// Syscall id to be attached. -1 for all syscalls
	int sys_nr;
	// True for syscall entry, false for syscall exit
	bool is_enter;
	// Initializa this private data instance from the string format of
	// tracepoint id
	int initialize_from_string(const std::string_view &sv);
};
} // namespace attach
} // namespace bpftime

#endif
