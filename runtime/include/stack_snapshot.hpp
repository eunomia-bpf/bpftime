// SPDX-License-Identifier: MIT
// Fault-safe bounded stack snapshot for userspace probe reads.
//
// In rootless mode BPF programs run inside the traced process itself, so the
// kernel's copy_from_user_nofault() fault boundary does not exist. The default
// fault-safe backend therefore issues one process_vm_readv() per bpf_probe_read
// (two syscalls per unwound frame), which dominates the cost of a userspace
// event on deep stacks.
//
// This backend captures a single, bounded window of the live stack once per
// BPF program invocation. Probe reads fully contained in that window are
// served from the snapshot with integer bounds checking; every other read
// (another mapping, a frame above the window, a stale or unmapped pointer)
// falls through to the compiled-in backend, whose process_vm_readv() still
// converts bad addresses into -EFAULT. The snapshot is therefore a pure
// acceleration layer: it never changes the observable result of a probe read,
// and a failed capture makes every read use the legacy backend unchanged.
//
// Build: ENABLE_STACK_SNAPSHOT=1 (CMake option, Linux x86_64/aarch64).
// Runtime opt-in: BPFTIME_EXPERIMENT_STACK_SNAPSHOT=1. When the option is off
// at compile time, the env is unset, or the fault-safe capture fails, every
// helper read behaves exactly as before.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) && \
	defined(BPFTIME_PROBE_STACK_SNAPSHOT)
#define BPFTIME_HAVE_STACK_SNAPSHOT 1
#else
#define BPFTIME_HAVE_STACK_SNAPSHOT 0
#endif

namespace bpftime::stack_snapshot
{

// Maximum captured stack window. The memaccess benchmark (see
// docs/memaccess-perf-report) shows the capture cost is dominated by the
// syscall fixed cost up to 4 KiB; larger windows grow linearly and explode
// under multi-threaded lock contention.
constexpr size_t capture_cap_bytes = 4096;

struct replay_context {
	// Readable mapping that contained the sampled SP: [stack_begin, stack_end).
	uintptr_t stack_begin = 0;
	uintptr_t stack_end = 0;
	// Lowest address captured; the window covers [sampled_sp, sampled_sp + window.size()).
	uintptr_t sampled_sp = 0;
	std::span<const std::byte> window;
};

// Non-null only while a stack_sample_capture is active on this thread.
extern thread_local const replay_context *current_sample;

// Runtime gate. The environment is read on every event entry rather than
// cached: getenv() is negligible next to a syscall, and it keeps the behavior
// deterministic for tests and for processes re-execing with a changed env.
bool enabled();

// Result of attempting to serve a probe read from the active snapshot.
enum class snapshot_read_result {
	// The read is not fully inside the captured window. The caller must
	// use the compiled-in probe read backend so the result (success or
	// fault-safe -EFAULT) stays identical to a build without snapshots.
	fallback,
	// The bytes were copied from the bounded snapshot.
	ok,
};

// Bounds-checked snapshot lookup. Never touches the kernel; safe to call from
// the BPF helper hot path.
snapshot_read_result try_read(void *dst, uintptr_t ptr, size_t size);

// RAII per-event capture. One object lives on the stack of one
// bpftime_prog::bpftime_prog_exec() call; nested invocations restore the
// outer context on destruction.
class stack_sample_capture
{
      public:
	// sp == 0 captures from the current hardware stack pointer. An
	// explicit SP is used by tests and by paths that already hold the
	// hooked thread's pt_regs.
	explicit stack_sample_capture(uintptr_t sp = 0) noexcept;
	~stack_sample_capture();
	stack_sample_capture(const stack_sample_capture &) = delete;
	stack_sample_capture &
	operator=(const stack_sample_capture &) = delete;

	bool active() const
	{
		return is_active;
	}

      private:
	// Fixed-size member storage keeps the capture heap-allocation free
	// (the BPF program may run inside a malloc interposer) and makes
	// nested captures independent. One live capture per nesting level
	// costs at most capture_cap_bytes of real stack.
	std::array<std::byte, capture_cap_bytes> buffer{};
	replay_context context;
	const replay_context *previous = nullptr;
	size_t captured_length = 0;
	bool is_active = false;
};

} // namespace bpftime::stack_snapshot
