#ifndef _BPFTIME_SYSCALL_TRACE_ATTACH_IMPL_HPP
#define _BPFTIME_SYSCALL_TRACE_ATTACH_IMPL_HPP
#include <base_attach_impl.hpp>
#include <memory>
#include <set>
#include <unordered_map>
namespace bpftime
{
namespace attach
{
// Represent the syscall hooker function
using syscall_hooker_func_t = int64_t (*)(int64_t sys_nr, int64_t arg1,
					  int64_t arg2, int64_t arg3,
					  int64_t arg4, int64_t arg5,
					  int64_t arg6);

struct trace_entry {
	short unsigned int type;
	unsigned char flags;
	unsigned char preempt_count;
	int pid;
};
// Used for ebpf arguments
struct trace_event_raw_sys_enter {
	struct trace_entry ent;
	long int id;
	long unsigned int args[6];
	char __data[0];
};
// Used for ebpf arguments
struct trace_event_raw_sys_exit {
	struct trace_entry ent;
	long int id;
	long int ret;
	char __data[0];
};

// Attach type id of syscall trace
constexpr size_t ATTACH_SYSCALL_TRACE = 2;

// An attach entry of syscall trace
struct syscall_trace_attach_entry {
	ebpf_run_callback cb;
	int sys_nr;
	bool is_enter;
};
// The global syscall trace instance. This one could be accessed by text segment
// transformer
extern std::optional<class syscall_trace_attach_impl *>
	global_syscall_trace_attach_impl;

// Re-entrancy depth for syscall dispatch. A userspace program that attaches
// its own tracepoint issues bpf()/tracepoint syscalls from inside the process
// whose syscalls are already dispatched; this thread-local depth lets the
// nested dispatch skip the callbacks and run only the original syscall.
//
// It is an `inline thread_local` with the initial-exec TLS model, identical to
// current_signal_callback/signal_handler_depth in base_attach_impl.hpp. That
// folds to a direct %fs (static-TLS) access at link time, so the per-syscall
// hot path makes no __tls_get_addr call. A dynamic-TLS call at this frequency
// (the bpftrace raw_syscalls tracepoint on a tight read/write loop) crashes
// inside the dlmopen link namespace used by the text segment transformer.
// initial-exec is proven safe on this exact load path by the two existing
// base_attach_impl variables that ship in the same agent shared object.
#if defined(__linux__)
__attribute__((tls_model("initial-exec")))
#endif
inline thread_local int syscall_dispatch_depth = 0;

// Used by text segment transformer to setup syscall callback
// Text segment transformer should provide a pointer to its syscall executor
// function. syscall trace attach impl will save the original syscall function,
// and replace it with one that was handled by syscall trace attach impl
extern "C" void
_bpftime__setup_syscall_hooker_callback(syscall_hooker_func_t *hooker);

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class syscall_trace_attach_impl final : public base_attach_impl {
    public:
	// Dispatch a syscall from text transformer
	int64_t dispatch_syscall(int64_t sys_nr, int64_t arg1, int64_t arg2,
				 int64_t arg3, int64_t arg4, int64_t arg5,
				 int64_t arg6);
	// Set the function of calling original syscall
	void set_original_syscall_function(syscall_hooker_func_t func)
	{
		orig_syscall = func;
	}
	// Set this syscall trace attach impl instance to the global ones, which
	// could be accessed by text segment transformer
	void set_to_global()
	{
		global_syscall_trace_attach_impl = this;
	}
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	syscall_trace_attach_impl(const syscall_trace_attach_impl &) = delete;
	syscall_trace_attach_impl &
	operator=(const syscall_trace_attach_impl &) = delete;
	syscall_trace_attach_impl()
	{
	}

    private:
	// The original syscall function
	syscall_hooker_func_t orig_syscall = nullptr;
	std::set<syscall_trace_attach_entry *> global_enter_callbacks,
		global_exit_callbacks;
	std::set<syscall_trace_attach_entry *> sys_enter_callbacks[512],
		sys_exit_callbacks[512];
	std::unordered_map<int, std::unique_ptr<syscall_trace_attach_entry> >
		attach_entries;
};

} // namespace attach
} // namespace bpftime
#endif
