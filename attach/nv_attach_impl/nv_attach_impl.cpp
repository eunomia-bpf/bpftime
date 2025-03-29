#include "nv_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <optional>

#ifdef __linux__
#include <asm/unistd.h> // For architecture-specific syscall numbers
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <cstdio>
#include <cstring>


// ------------------------------------------------------------------
// Implementation of the nv_attach_impl interface
// ------------------------------------------------------------------
namespace bpftime
{
namespace attach
{

// Optionally keep a global pointer if needed:
std::optional<nv_attach_impl *> global_nv_attach_impl;

// We’ll hold a unique_ptr to our CUDAInjector in nv_attach_impl
// so we can attach/detach by ID (PID) and store state.
struct nv_attach_impl::Impl {
	std::unique_ptr<CUDAInjector> injector;
};

//
// Give nv_attach_impl a private pImpl pointer, so we only
// expose the class interface in the header.
//
static inline std::unique_ptr<nv_attach_impl::Impl> &
get_impl(nv_attach_impl &obj)
{
	static std::unique_ptr<nv_attach_impl::Impl> s_impl;
	if (!s_impl) {
		s_impl = std::make_unique<nv_attach_impl::Impl>();
	}
	return s_impl;
}

int64_t nv_attach_impl::dispatch_nv(int64_t arg1, int64_t arg2, int64_t arg3,
				    int64_t arg4, int64_t arg5, int64_t arg6)
{
	spdlog::debug(
		"nv_attach_impl::dispatch_syscall({}, {}, {}, {}, {}, {})",
		arg1, arg2, arg3, arg4, arg5, arg6);

	// For now, just returning 0 or an error code as a placeholder.
	// Real logic might do PTRACE_SYSCALL on the attached process, etc.
	return 0;
}

int nv_attach_impl::detach_by_id(int id)
{
	// Suppose `id` is actually the PID we attached. We can reuse the same
	// pImpl.
	auto &implPtr = get_impl(*this);
	if (!implPtr->injector) {
		spdlog::warn("detach_by_id({}): no active injector!", id);
		return -1;
	}

	spdlog::info("nv_attach_impl::detach_by_id({})", id);
	if (!implPtr->injector->detach()) {
		spdlog::error("Failed to detach from PID {}", id);
		return -1;
	}

	spdlog::info("Detached from PID {}", id);
	implPtr->injector.reset();
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	spdlog::info(
		"nv_attach_impl::create_attach_with_ebpf_callback(...) attach_type={}",
		attach_type);

	// For example, we interpret private_data.pid as the target PID:
	pid_t target_pid =
		static_cast<const nv_attach_private_data &>(private_data).pid;

	// Create our injector if not yet created
	auto &implPtr = get_impl(*this);
	implPtr->injector = std::make_unique<CUDAInjector>(target_pid);

	// Attempt to attach
	if (!implPtr->injector->attach()) {
		spdlog::error("Failed to attach to PID {}", target_pid);
		return -1;
	}

	// If we have an eBPF callback, call it here
	if (cb) {
		// For example:
		void *arg1 = nullptr;
		unsigned long arg2 = 0;
		unsigned long *arg3 = nullptr;

		int rc = cb(arg1, arg2, arg3);
		spdlog::info("ebpf_run_callback returned {}", rc);
	}

	return 0;
}

// A C interface for hooking up your own system-call hooking logic
extern "C" void _bpftime__setup_nv_hooker_callback(nv_hooker_func_t *hooker)
{
	spdlog::info("_bpftime__setup_nv_hooker_callback(...) called");
	// Example: store the callback pointer if you want
	// In real usage, you might store it globally or pass it somewhere else.
}

int nv_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	spdlog::info("nv_attach_private_data::initialize_from_string({})", sv);
    
	return 0;
}

} // namespace attach
} // namespace bpftime
