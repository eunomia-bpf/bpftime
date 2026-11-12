#ifndef _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_SYSCALL_TRACE_ATTACH_PRIVATE_DATA_HPP
#include <attach_private_data.hpp>
#include <cstdint>
namespace bpftime
{
namespace attach
{
enum class syscall_kprobe_abi : uint8_t {
	syscall_wrapper,
	do_mmap,
	do_munmap,
};

// Private data of syscall trace attach
struct syscall_trace_attach_private_data : public attach_private_data {
	// Syscall id to be attached. -1 for all syscalls
	int sys_nr;
	// True for syscall entry, false for syscall exit
	bool is_enter;
	// Kernel functions such as do_mmap use the normal function-call ABI,
	// unlike __x64_sys_mmap which receives a nested pt_regs pointer.
	syscall_kprobe_abi kprobe_abi = syscall_kprobe_abi::syscall_wrapper;
	// Initializa this private data instance from the string format of
	// tracepoint id
	int initialize_from_string(const std::string_view &sv);
	std::string to_string() const override
	{
		return std::to_string(sys_nr) + (is_enter ? ":enter" : ":exit");
	}
};
} // namespace attach
} // namespace bpftime

#endif
