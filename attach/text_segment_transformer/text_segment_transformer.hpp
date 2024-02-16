#ifndef _TEST_SEGMENT_TRANSFORMER_H
#define _TEST_SEGMENT_TRANSFORMER_H
// C++ standard library could not be found by clangd on my machine.. Will fix in
// a later time
#include <cinttypes>
using syscall_hooker_func_t = int64_t (*)(int64_t sys_nr, int64_t arg1,
					  int64_t arg2, int64_t arg3,
					  int64_t arg4, int64_t arg5,
					  int64_t arg6);
namespace bpftime
{
	// Setup userspace syscall trace
void setup_syscall_tracer();
// Get current callback function when a syscall was invoked. Default to be a function that directly calls the syscall
syscall_hooker_func_t get_call_hook();
// Set the syscall callback function
void set_call_hook(syscall_hooker_func_t hook);
} // namespace bpftime

#endif
