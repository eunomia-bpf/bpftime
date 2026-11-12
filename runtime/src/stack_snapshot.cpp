// SPDX-License-Identifier: MIT
#include "stack_snapshot.hpp"

#if BPFTIME_HAVE_STACK_SNAPSHOT

#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

namespace bpftime::stack_snapshot
{

thread_local const replay_context *current_sample = nullptr;

namespace
{

struct stack_interval {
	uintptr_t begin = 0;
	uintptr_t end = 0;
};

// O(1) on the hot path: a hit is two integer comparisons. The cache is only
// rebuilt from /proc/self/maps when the SP leaves the cached mapping, which
// happens on thread/fiber stack switches, not on ordinary frame unwinding.
thread_local stack_interval cached_interval;

bool resolve_stack_mapping(uintptr_t sp, stack_interval &out)
{
	if (sp >= cached_interval.begin && sp < cached_interval.end) {
		out = cached_interval;
		return true;
	}

	std::ifstream maps("/proc/self/maps");
	if (!maps.is_open())
		return false;

	std::string line;
	while (std::getline(maps, line)) {
		uintptr_t low = 0;
		uintptr_t high = 0;
		char perms[5] = {};
		if (std::sscanf(line.c_str(), "%lx-%lx %4s", &low, &high,
				perms) != 3)
			continue;
		if (perms[0] == 'r' && sp >= low && sp < high) {
			cached_interval = { low, high };
			out = cached_interval;
			return true;
		}
	}
	return false;
}

// Userspace equivalent of copy_from_user_nofault(): the kernel resolves the
// fault and returns EFAULT instead of delivering a signal.
bool fault_safe_read(void *dst, const void *src, size_t size)
{
	struct iovec local {
		dst, size
	};
	struct iovec remote {
		const_cast<void *>(src), size
	};
	return syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1,
		       0) == static_cast<ssize_t>(size);
}

uintptr_t current_stack_pointer()
{
#if defined(__x86_64__)
	register uintptr_t sp asm("rsp");
	asm volatile("" : "=r"(sp));
	return sp;
#elif defined(__aarch64__)
	register uintptr_t sp asm("sp");
	asm volatile("" : "=r"(sp));
	return sp;
#else
#error "Unsupported architecture for stack snapshot"
#endif
}

} // namespace

bool enabled()
{
	const char *value = getenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
	return value != nullptr && *value != '\0' && *value != '0';
}

snapshot_read_result try_read(void *dst, uintptr_t ptr, size_t size)
{
	const replay_context *sample = current_sample;
	if (sample == nullptr || size == 0)
		return snapshot_read_result::fallback;

	// Anything outside the stack mapping goes through the compiled-in
	// backend: heap allocations, CUDA buffers and other mappings keep
	// their exact previous semantics.
	if (ptr < sample->stack_begin || ptr >= sample->stack_end)
		return snapshot_read_result::fallback;

	// D-gate: the address belongs to a stack mapping but cannot be served
	// from the captured window. A stale frame pointer saved across a fiber
	// switch lands here and is rejected with no syscall at all.
	if (ptr < sample->sampled_sp)
		return snapshot_read_result::fault;

	const uintptr_t offset = ptr - sample->sampled_sp;
	if (offset > sample->window.size() ||
	    size > sample->window.size() - offset)
		return snapshot_read_result::fault;

	std::memcpy(dst, sample->window.data() + offset, size);
	return snapshot_read_result::ok;
}

stack_sample_capture::stack_sample_capture(uintptr_t target_sp) noexcept
{
	if (!enabled())
		return;

	const uintptr_t sp = target_sp != 0 ? target_sp :
					      current_stack_pointer();

	stack_interval interval;
	if (!resolve_stack_mapping(sp, interval))
		return;

	const size_t available = interval.end - sp;
	captured_length =
		available < capture_cap_bytes ? available : capture_cap_bytes;
	if (captured_length == 0)
		return;

	if (!fault_safe_read(buffer.data(),
			     reinterpret_cast<const void *>(sp),
			     captured_length)) {
		SPDLOG_DEBUG("stack snapshot capture failed at sp={:x}, "
			     "falling back to per-read probe backend",
			     sp);
		captured_length = 0;
		return;
	}

	context.stack_begin = interval.begin;
	context.stack_end = interval.end;
	context.sampled_sp = sp;
	context.window = std::as_bytes(
		std::span(buffer.data(), captured_length));
	previous = current_sample;
	current_sample = &context;
	is_active = true;
}

stack_sample_capture::~stack_sample_capture()
{
	if (is_active)
		current_sample = previous;
}

} // namespace bpftime::stack_snapshot

#endif
