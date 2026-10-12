/* SPDX-License-Identifier: MIT
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_helper_read_and_write.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <sys/uio.h>
#endif

// Keep this translation unit independent of libbpf and linux/bpf.h: the
// Frida devkit contains Capstone's own bpf_insn type.
#if defined(BPFTIME_HAVE_FRIDA) && defined(__linux__) &&                       \
	(defined(__x86_64__) || defined(__aarch64__))
#define BPFTIME_SIGNAL_PROBE_ACCESS 1
#include <frida-gum.h>
#include <setjmp.h>
#include <signal.h>
#include <ucontext.h>
#endif

namespace
{
#if defined(__linux__) && (defined(ENABLE_PROBE_READ_CHECK) ||                 \
			   defined(ENABLE_PROBE_WRITE_CHECK))
static pid_t probe_access_pid = getpid();
static const int probe_access_atfork_status =
	pthread_atfork(nullptr, nullptr, [] { probe_access_pid = getpid(); });
#endif

#ifdef BPFTIME_SIGNAL_PROBE_ACCESS
// Process-lifetime objects: a registered handler must never observe a freed
// backtracer. This helper is deliberately NOT audited for SIGTRAP callbacks.
static std::once_flag probe_access_once;
static GumBacktracer *probe_access_backtracer = nullptr;
static bool probe_access_handler_ready = false;
static thread_local sigjmp_buf probe_access_jump;
static thread_local sigset_t probe_access_fault_mask;
static thread_local volatile sig_atomic_t probe_access_active = 0;
extern "C" const char bpftime_probe_copy_begin[], bpftime_probe_copy_end[];

static void probe_access_sigsegv(int sig, siginfo_t *info, void *context)
{
	if (probe_access_active &&
	    (info->si_code == SEGV_MAPERR || info->si_code == SEGV_ACCERR)) {
		// A fault in the unwinder itself must not recursively attempt
		// recovery.
		probe_access_active = 0;
		auto *uc = static_cast<ucontext_t *>(context);
		GumCpuContext cpu = {};
#if defined(__x86_64__)
#define COPY_GREG(field, reg) cpu.field = uc->uc_mcontext.gregs[reg]
		COPY_GREG(rip, REG_RIP);
		COPY_GREG(rsp, REG_RSP);
		COPY_GREG(rbp, REG_RBP);
		COPY_GREG(rax, REG_RAX);
		COPY_GREG(rbx, REG_RBX);
		COPY_GREG(rcx, REG_RCX);
		COPY_GREG(rdx, REG_RDX);
		COPY_GREG(rdi, REG_RDI);
		COPY_GREG(rsi, REG_RSI);
		COPY_GREG(r8, REG_R8);
		COPY_GREG(r9, REG_R9);
		COPY_GREG(r10, REG_R10);
		COPY_GREG(r11, REG_R11);
		COPY_GREG(r12, REG_R12);
		COPY_GREG(r13, REG_R13);
		COPY_GREG(r14, REG_R14);
		COPY_GREG(r15, REG_R15);
#undef COPY_GREG
#elif defined(__aarch64__)
		cpu.pc = uc->uc_mcontext.pc;
		cpu.sp = uc->uc_mcontext.sp;
		cpu.nzcv = uc->uc_mcontext.pstate;
		for (unsigned i = 0; i < 29; ++i)
			cpu.x[i] = uc->uc_mcontext.regs[i];
		cpu.fp = uc->uc_mcontext.regs[29];
		cpu.lr = uc->uc_mcontext.regs[30];
#endif
		GumReturnAddressArray frames = {};
		gum_backtracer_generate(probe_access_backtracer, &cpu, &frames);
		for (guint i = 0; i < frames.len; ++i) {
			auto address =
				reinterpret_cast<uintptr_t>(frames.items[i]);
			if (address >= reinterpret_cast<uintptr_t>(
					       bpftime_probe_copy_begin) &&
			    address <= reinterpret_cast<uintptr_t>(
					       bpftime_probe_copy_end)) {
				probe_access_fault_mask = uc->uc_sigmask;
				siglongjmp(probe_access_jump, 1);
			}
		}
	}

	struct sigaction action = {};
	action.sa_handler = SIG_DFL;
	sigemptyset(&action.sa_mask);
	sigaction(sig, &action, nullptr);
	raise(sig);
}

static bool initialize_probe_access_handler()
{
	// Check only during initialization. Later host changes to signal masks
	// or disposition cannot be detected without a syscall on every copy.
	sigset_t mask;
	if (pthread_sigmask(SIG_SETMASK, nullptr, &mask) != 0 ||
	    sigismember(&mask, SIGSEGV) != 0)
		return false;
	gum_init_embedded();
	probe_access_backtracer = gum_backtracer_make_accurate();
	if (!probe_access_backtracer) {
		return false;
	}
	struct sigaction action = {};
	action.sa_sigaction = probe_access_sigsegv;
	action.sa_flags = SA_SIGINFO;
	sigemptyset(&action.sa_mask);
	probe_access_handler_ready = sigaction(SIGSEGV, &action, nullptr) == 0;
	if (!probe_access_handler_ready) {
		return false;
	}
	return true;
}
#endif

#ifdef BPFTIME_SIGNAL_PROBE_ACCESS
enum class protected_copy_result { success, fault, unavailable };

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noclone))
#endif
__attribute__((noinline)) static protected_copy_result
signal_protected_copy(void *dst, const void *src, size_t size)
{
	if (probe_access_active)
		return protected_copy_result::unavailable;
	int saved_errno = errno;
	try {
		std::call_once(probe_access_once,
			       initialize_probe_access_handler);
	} catch (...) {
		// Initialization failures must not unwind through a BPF VM or
		// host.
		errno = saved_errno;
		return protected_copy_result::unavailable;
	}
	errno = saved_errno;
	if (!probe_access_handler_ready)
		return protected_copy_result::unavailable;
	if (sigsetjmp(probe_access_jump, 0) != 0) {
		// siglongjmp bypasses sigreturn, leaving SIGSEGV blocked.
		// Restore the fault-time mask only here; successful copies need
		// no syscall.
		pthread_sigmask(SIG_SETMASK, &probe_access_fault_mask, nullptr);
		probe_access_active = 0;
		errno = saved_errno;
		return protected_copy_result::fault;
	}
	probe_access_active = 1;
	// Both read and write helpers reach this frame. Match the memcpy return
	// address without resolving symbols or allocating in the handler.
	void *(*volatile copy)(void *, const void *, size_t) = memcpy;
	asm volatile(".local bpftime_probe_copy_begin\n"
		     "bpftime_probe_copy_begin:" ::
			     : "memory");
	copy(dst, src, size);
	asm volatile(".local bpftime_probe_copy_end\n"
		     "bpftime_probe_copy_end:" ::
			     : "memory");
	probe_access_active = 0;
	errno = saved_errno;
	return protected_copy_result::success;
}
#endif

enum class access_direction { read, write };
enum class access_backend { direct, checked, signal };

#ifdef ENABLE_PROBE_READ_CHECK
static constexpr auto read_backend = access_backend::checked;
#else
static constexpr auto read_backend = access_backend::direct;
#endif
#ifdef ENABLE_PROBE_WRITE_CHECK
static constexpr auto write_backend = access_backend::checked;
#else
static constexpr auto write_backend = access_backend::direct;
#endif

// Common interface for fixed-size reads, writes, and string read/clear
// operations. Direction only matters to the platform syscall backend;
// SIGSEGV recovery and memcpy use the same implementation for both.
static bool probe_access_memory(void *dst, const void *src, size_t size,
				[[maybe_unused]] access_direction direction,
				access_backend backend)
{
	if (size == 0)
		return true;
#ifdef BPFTIME_SIGNAL_PROBE_ACCESS
	if (backend == access_backend::signal) {
		auto result = signal_protected_copy(dst, src, size);
		if (result != protected_copy_result::unavailable)
			return result == protected_copy_result::success;
		// Only protection setup failures fall back. A recovered memory
		// fault must never retry the same invalid copy without
		// protection.
		backend = read_backend;
		if (direction == access_direction::write)
			backend = write_backend;
	}
#endif
	if (backend == access_backend::direct) {
		memcpy(dst, src, size);
		return true;
	}
#if defined(ENABLE_PROBE_READ_CHECK) || defined(ENABLE_PROBE_WRITE_CHECK)
#if defined(__APPLE__)
	mach_vm_size_t copied = 0;
	return mach_vm_read_overwrite(mach_task_self(), (mach_vm_address_t)src,
				      (mach_vm_size_t)size,
				      (mach_vm_address_t)dst,
				      &copied) == KERN_SUCCESS &&
	       copied == size;
#elif defined(__linux__)
	struct iovec local = { direction == access_direction::read ?
				       dst :
				       const_cast<void *>(src),
			       size };
	struct iovec remote = { direction == access_direction::read ?
					const_cast<void *>(src) :
					dst,
				size };
	auto number = direction == access_direction::read ?
			      SYS_process_vm_readv :
			      SYS_process_vm_writev;
	auto pid =
		probe_access_atfork_status == 0 ? probe_access_pid : getpid();
	return syscall(number, pid, &local, 1, &remote, 1, 0) == (ssize_t)size;
#else
#error "Probe memory checks are unsupported on this platform"
#endif
#else
	return false;
#endif
}

static bool probe_read_memory(void *dst, const void *src, size_t size,
			      access_backend backend = read_backend)
{
	return probe_access_memory(dst, src, size, access_direction::read,
				   backend);
}

static bool probe_write_memory(void *dst, const void *src, size_t size,
			       access_backend backend = write_backend)
{
	return probe_access_memory(dst, src, size, access_direction::write,
				   backend);
}

#ifdef BPFTIME_SIGNAL_PROBE_ACCESS
static constexpr auto user_read_backend = access_backend::signal;
#ifndef ENABLE_PROBE_WRITE_CHECK
static constexpr auto user_write_backend = access_backend::signal;
#else
// Retain the explicitly configured platform write checks. Otherwise user
// writes use the same signal-protected copy as user reads.
static constexpr auto user_write_backend = write_backend;
#endif
#else
static constexpr auto user_read_backend = read_backend;
static constexpr auto user_write_backend = write_backend;
#endif

#ifdef ENABLE_PROBE_READ_CHECK
static size_t probe_page_remaining(const void *address)
{
	static const size_t page_size = []() {
		long result = sysconf(_SC_PAGESIZE);
		return result > 0 ? (size_t)result : (size_t)4096;
	}();
	return page_size - ((uintptr_t)address % page_size);
}

static void clear_probe_memory(void *dst, size_t size)
{
	static const char zeros[256] = {};
	auto output = (char *)dst;
	while (size > 0) {
		size_t chunk_size = std::min(size, sizeof(zeros));
		if (!probe_write_memory(output, zeros, chunk_size,
					access_backend::checked))
			return;
		output += chunk_size;
		size -= chunk_size;
	}
}
#endif
} // namespace

extern "C" {
int64_t bpftime_probe_read(uint64_t dst, int64_t size, uint64_t src, uint64_t,
			   uint64_t)
{
	if (size < 0)
		return -EFAULT;
	return probe_read_memory((void *)dst, (const void *)src, (size_t)size) ?
		       0 :
		       -EFAULT;
}

int64_t bpftime_probe_read_user(uint64_t dst, int64_t size, uint64_t src,
				uint64_t, uint64_t)
{
	if (size < 0)
		return -EFAULT;
	return probe_read_memory((void *)dst, (const void *)src, (size_t)size,
				 user_read_backend) ?
		       0 :
		       -EFAULT;
}

int64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, int64_t size,
				 uint64_t, uint64_t)
{
	if (size < 0)
		return -EFAULT;
	return probe_write_memory((void *)dst, (const void *)src, (size_t)size,
				  user_write_backend) ?
		       0 :
		       -EFAULT;
}

int64_t bpf_probe_read_str(uint64_t buf, uint64_t bufsz, uint64_t ptr, uint64_t,
			   uint64_t)
{
	if (bufsz == 0)
		return 0;

	auto dst = (char *)(uintptr_t)buf;
	auto src = (const char *)(uintptr_t)ptr;
#ifndef ENABLE_PROBE_READ_CHECK
	for (size_t i = 0; i < (size_t)bufsz; ++i) {
		char value;
		if (!probe_read_memory(&value, src + i, 1))
			return -EFAULT;
		bool complete = value == '\0' || i == (size_t)bufsz - 1;
		if (complete)
			value = '\0';
		if (!probe_write_memory(dst + i, &value, 1,
					access_backend::direct))
			return -EFAULT;
		if (complete)
			return (int64_t)i + 1;
	}
	return (int64_t)bufsz;
#else
	char scratch[256];
	size_t copied = 0;
	while (copied < (size_t)bufsz) {
		size_t chunk_size =
			std::min(sizeof(scratch), (size_t)bufsz - copied);
		chunk_size = std::min(chunk_size,
				      probe_page_remaining(src + copied));
		if (!probe_read_memory(scratch, src + copied, chunk_size)) {
			clear_probe_memory(dst, (size_t)bufsz);
			return -EFAULT;
		}

		auto terminator = (char *)memchr(scratch, '\0', chunk_size);
		bool complete = terminator != nullptr;
		if (complete) {
			chunk_size = (size_t)(terminator - scratch) + 1;
		} else if (chunk_size == (size_t)bufsz - copied) {
			scratch[chunk_size - 1] = '\0';
			complete = true;
		}

		if (!probe_write_memory(dst + copied, scratch, chunk_size,
					access_backend::checked))
			return -EFAULT;
		copied += chunk_size;
		if (complete)
			return (int64_t)copied;
	}
	return (int64_t)copied;
#endif
}

} // extern "C"
