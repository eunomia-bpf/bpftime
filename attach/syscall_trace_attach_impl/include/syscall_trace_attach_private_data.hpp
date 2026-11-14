#ifndef _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#include <attach_private_data.hpp>
#include <cstdint>
namespace bpftime
{
namespace attach
{
// Private data of syscall trace attach
struct syscall_trace_attach_private_data : public attach_private_data {
	// Kernel trace_entry IDs are 16-bit. Reserve a disjoint positive range
	// for rootless IDs, decoded before consulting tracefs or callback arrays.
	static constexpr int32_t synthetic_id_base = 0x40000000;
	static int32_t synthetic_id_for_event(const std::string_view &event);
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
