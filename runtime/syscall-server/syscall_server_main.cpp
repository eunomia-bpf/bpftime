/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "syscall_context.hpp"
#if defined(__x86_64__)
#include <asm/unistd_64.h>
#else
#include <asm-generic/unistd.h>
#endif
#include <boost/interprocess/exceptions.hpp>
#include <cstdio>
#if __linux__
#include "linux/bpf.h"
#include <asm-generic/errno-base.h>
#endif
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <atomic>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <spdlog/cfg/env.h>
#include <cstdarg>
#include <sched.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <utility>

// 0 = uninitialized, 1 = in progress, 2 = done, 3 = disabled
static int ctx_initialized = 0;
static __thread int tls_initializing = 0;

// Helper function for safe logging with pointer parameters
inline const char *safe_ptr_str(const char *ptr)
{
	return ptr ? ptr : "<null>";
}

// Safe debug logging that checks if logger is initialized
// This prevents crashes during logger initialization (e.g., when fopen is
// called by spdlog itself)
template <typename... Args>
inline void safe_spdlog_debug(spdlog::format_string_t<Args...> fmt,
			      Args &&...args) noexcept
{
	if (spdlog::default_logger_raw()) {
		try {
			spdlog::debug(fmt, std::forward<Args>(args)...);
		} catch (...) {
		}
	}
}

template <typename... Args>
inline void safe_spdlog_error(spdlog::format_string_t<Args...> fmt,
			      Args &&...args) noexcept
{
	if (__atomic_load_n(&ctx_initialized, __ATOMIC_ACQUIRE) == 2 &&
	    spdlog::default_logger_raw()) {
		try {
			spdlog::error(fmt, std::forward<Args>(args)...);
		} catch (...) {
		}
	}
}

template <typename T> static T resolve_next_symbol(const char *name) noexcept
{
	return (T)dlsym(RTLD_NEXT, name);
}

using raw_syscall_fn = long (*)(long, ...);
using close_fn = int (*)(int);
using mmap64_fn = void *(*)(void *, size_t, int, int, int, off64_t);
using mmap_fn = void *(*)(void *, size_t, int, int, int, off_t);
using ioctl_fn = int (*)(int, unsigned long, ...);
using epoll_create1_fn = int (*)(int);
using epoll_ctl_fn = int (*)(int, int, int, epoll_event *);
using epoll_wait_fn = int (*)(int, epoll_event *, int, int);
using munmap_fn = int (*)(void *, size_t);
using openat_fn = int (*)(int, const char *, int, ...);
using open_fn = int (*)(const char *, int, ...);
using read_fn = ssize_t (*)(int, void *, size_t);
using fopen_fn = FILE *(*)(const char *, const char *);

template <typename Fn> struct cached_next_symbol {
	std::atomic<Fn> value{ nullptr };
	std::atomic<bool> initialized{ false };
};

static cached_next_symbol<raw_syscall_fn> next_syscall_symbol;
static cached_next_symbol<close_fn> next_close_symbol;
static cached_next_symbol<mmap64_fn> next_mmap64_symbol;
static cached_next_symbol<mmap_fn> next_mmap_symbol;
static cached_next_symbol<ioctl_fn> next_ioctl_symbol;
static cached_next_symbol<epoll_create1_fn> next_epoll_create1_symbol;
static cached_next_symbol<epoll_ctl_fn> next_epoll_ctl_symbol;
static cached_next_symbol<epoll_wait_fn> next_epoll_wait_symbol;
static cached_next_symbol<munmap_fn> next_munmap_symbol;
static cached_next_symbol<openat_fn> next_openat_symbol;
static cached_next_symbol<open_fn> next_open_symbol;
static cached_next_symbol<read_fn> next_read_symbol;
static cached_next_symbol<fopen_fn> next_fopen_symbol;
static cached_next_symbol<fopen_fn> next_fopen64_symbol;

static bool open_flags_have_mode(int flags) noexcept
{
	if ((flags & O_CREAT) != 0)
		return true;
#ifdef O_TMPFILE
	if ((flags & O_TMPFILE) == O_TMPFILE)
		return true;
#endif
	return false;
}

template <typename Fn>
static Fn resolve_cached_next_symbol(cached_next_symbol<Fn> &symbol,
				     const char *name) noexcept
{
	if (!symbol.initialized.load(std::memory_order_acquire)) {
		auto fn = resolve_next_symbol<Fn>(name);
		symbol.value.store(fn, std::memory_order_release);
		symbol.initialized.store(true, std::memory_order_release);
		return fn;
	}
	return symbol.value.load(std::memory_order_acquire);
}

template <typename Fn, typename Ret, typename... Args>
static Ret call_next_symbol(cached_next_symbol<Fn> &symbol, const char *name,
			    Ret failure,
			    Args... args) noexcept
{
	auto fn = resolve_cached_next_symbol(symbol, name);
	if (!fn) {
		errno = ENOSYS;
		return failure;
	}
	return fn(args...);
}

static void *fallback_mmap64(void *addr, size_t length, int prot, int flags,
			     int fd, off64_t offset) noexcept
{
	auto fn64 = resolve_cached_next_symbol(next_mmap64_symbol, "mmap64");
	if (fn64)
		return fn64(addr, length, prot, flags, fd, offset);
	return call_next_symbol(next_mmap_symbol, "mmap", MAP_FAILED, addr,
				length, prot, flags, fd, (off_t)offset);
}

static int fallback_openat(int fd, const char *file, int oflag, mode_t mode,
			   bool has_mode) noexcept
{
	auto fn = resolve_cached_next_symbol(next_openat_symbol, "openat");
	if (!fn) {
		errno = ENOSYS;
		return -1;
	}
	return has_mode ? fn(fd, file, oflag, mode) : fn(fd, file, oflag);
}

static int fallback_open(const char *file, int oflag, mode_t mode,
			 bool has_mode) noexcept
{
	auto fn = resolve_cached_next_symbol(next_open_symbol, "open");
	if (!fn) {
		errno = ENOSYS;
		return -1;
	}
	return has_mode ? fn(file, oflag, mode) : fn(file, oflag);
}

static long fallback_syscall(long sysno, long arg1, long arg2, long arg3,
			     long arg4, long arg5, long arg6) noexcept
{
	auto fn = resolve_cached_next_symbol(next_syscall_symbol, "syscall");
	if (!fn) {
		errno = ENOSYS;
		return -1;
	}
	return fn(sysno, arg1, arg2, arg3, arg4, arg5, arg6);
}

// global context for bpf syscall server
union syscall_server_ctx_union {
	syscall_context ctx;
	syscall_context *operator->()
	{
		return &ctx;
	}
	syscall_server_ctx_union()
	{
	}
	~syscall_server_ctx_union()
	{
	}
};
static syscall_server_ctx_union context;
static bool initialize_ctx() noexcept
{
	if (tls_initializing)
		return false;
	int state = __atomic_load_n(&ctx_initialized, __ATOMIC_ACQUIRE);
	if (state == 2)
		return true;
	if (state == 3)
		return false;
	int expected = 0;
	if (__atomic_compare_exchange_n(&ctx_initialized, &expected, 1, false,
					__ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
		tls_initializing = 1;
		try {
			new (&context.ctx) syscall_context;
		} catch (...) {
			tls_initializing = 0;
			__atomic_store_n(&ctx_initialized, 3, __ATOMIC_RELEASE);
			return false;
		}
		tls_initializing = 0;
		__atomic_store_n(&ctx_initialized, 2, __ATOMIC_RELEASE);
		return true;
	} else {
		while ((state = __atomic_load_n(&ctx_initialized,
						__ATOMIC_ACQUIRE)) == 1) {
			sched_yield();
		}
		return state == 2;
	}
}

template <typename F, typename Fallback>
auto handle_exceptions(F &&f, Fallback &&fallback) noexcept -> decltype(f())
{
	try {
		return f();
	} catch (const boost::interprocess::bad_alloc &e) {
		safe_spdlog_error("Boost interprocess bad_alloc: {}", e.what());
		safe_spdlog_error(
			"Falling back to the original host operation");
		return fallback();
	} catch (const std::exception &e) {
		safe_spdlog_error("bpftime syscall interposer failed: {}",
				  e.what());
		return fallback();
	} catch (...) {
		safe_spdlog_error(
			"bpftime syscall interposer failed with an unknown exception");
		return fallback();
	}
}

template <typename F, typename Fallback>
auto call_with_context(F &&f, Fallback &&fallback) noexcept
	-> decltype(fallback())
{
	if (!initialize_ctx())
		return fallback();
	return handle_exceptions(std::forward<F>(f),
				 std::forward<Fallback>(fallback));
}

template <typename F, typename Fallback, typename... Args>
auto call_with_context_debug(spdlog::format_string_t<Args...> fmt, F &&f,
			     Fallback &&fallback, Args &&...args) noexcept
	-> decltype(fallback())
{
	return call_with_context(
		[&]() {
			safe_spdlog_debug(fmt, std::forward<Args>(args)...);
			return f();
		},
		std::forward<Fallback>(fallback));
}

extern "C" int epoll_wait(int epfd, epoll_event *evt, int maxevents,
			  int timeout)
{
	auto call_original = [&]() {
		return call_next_symbol(next_epoll_wait_symbol, "epoll_wait",
					-1, epfd, evt, maxevents, timeout);
	};
	return call_with_context_debug(
		"epoll_wait {}",
		[&]() { return context->handle_epoll_wait(epfd, evt, maxevents,
							  timeout); },
		call_original, epfd);
}

extern "C" int epoll_ctl(int epfd, int op, int fd, epoll_event *evt)
{
	auto call_original = [&]() {
		return call_next_symbol(next_epoll_ctl_symbol, "epoll_ctl",
					-1, epfd, op, fd, evt);
	};
	return call_with_context_debug(
		"epoll_ctl {} {} {} {}",
		[&]() { return context->handle_epoll_ctl(epfd, op, fd, evt); },
		call_original, epfd, op, fd, (uintptr_t)evt);
}

extern "C" int epoll_create1(int flags)
{
	auto call_original = [&]() {
		return call_next_symbol(next_epoll_create1_symbol,
					"epoll_create1", -1, flags);
	};
	return call_with_context_debug(
		"epoll_create1 {}",
		[&]() { return context->handle_epoll_create1(flags); },
		call_original, flags);
}

extern "C" int ioctl(int fd, unsigned long req, ...)
{
	va_list args;
	va_start(args, req);
	unsigned long arg3 = va_arg(args, long);
	va_end(args);
	auto call_original = [&]() {
		return call_next_symbol(next_ioctl_symbol, "ioctl", -1, fd,
					req, arg3);
	};
	return call_with_context_debug(
		"ioctl {} {} {}",
		[&]() { return context->handle_ioctl(fd, req, arg3); },
		call_original, fd, req, arg3);
}

extern "C" void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
			off64_t offset)
{
	auto call_original = [&]() {
		return fallback_mmap64(addr, length, prot, flags, fd, offset);
	};
	return call_with_context_debug(
		"mmap64 {:x}",
		[&]() { return context->handle_mmap64(addr, length, prot, flags,
						      fd, offset); },
		call_original, (uintptr_t)addr);
}

extern "C" void *mmap(void *addr, size_t length, int prot, int flags, int fd,
		      off_t offset)
{
	auto call_original = [&]() {
		return call_next_symbol(next_mmap_symbol, "mmap", MAP_FAILED,
					addr, length, prot, flags, fd,
					offset);
	};
	return call_with_context_debug(
		"mmap {:x}",
		[&]() { return context->handle_mmap(addr, length, prot, flags,
						    fd, offset); },
		call_original, (uintptr_t)addr);
}

extern "C" int munmap(void *addr, size_t size)
{
	auto call_original = [&]() {
		return call_next_symbol(next_munmap_symbol, "munmap", -1,
					addr, size);
	};
	return call_with_context_debug(
		"munmap {:x} {}",
		[&]() { return context->handle_munmap(addr, size); },
		call_original, (uintptr_t)addr, size);
}

extern "C" int close(int fd)
{
	auto call_original = [&]() {
		return call_next_symbol(next_close_symbol, "close", -1, fd);
	};
	return call_with_context_debug(
		"Closing fd {}",
		[&]() { return context->handle_close(fd); },
		call_original, fd);
}

extern "C" int openat(int fd, const char *file, int oflag, ...)
{
	bool has_mode = open_flags_have_mode(oflag);
	mode_t mode = 0;
	va_list args;
	va_start(args, oflag);
	if (has_mode)
		mode = va_arg(args, mode_t);
	va_end(args);
	auto call_original = [&]() {
		return fallback_openat(fd, file, oflag, mode, has_mode);
	};
	return call_with_context_debug(
		"openat {} {} {} {}",
		[&]() { return context->handle_openat(fd, file, oflag,
						      (unsigned short)mode); },
		call_original, fd, safe_ptr_str(file), oflag, mode);
}
extern "C" int open(const char *file, int oflag, ...)
{
	bool has_mode = open_flags_have_mode(oflag);
	mode_t mode = 0;
	va_list args;
	va_start(args, oflag);
	if (has_mode)
		mode = va_arg(args, mode_t);
	va_end(args);
	auto call_original = [&]() {
		return fallback_open(file, oflag, mode, has_mode);
	};
	return call_with_context_debug(
		"open {} {} {}",
		[&]() { return context->handle_open(file, oflag,
						    (unsigned short)mode); },
		call_original, safe_ptr_str(file), oflag, mode);
}
extern "C" ssize_t read(int fd, void *buf, size_t count)
{
	auto call_original = [&]() {
		return call_next_symbol(next_read_symbol, "read", (ssize_t)-1,
					fd, buf, count);
	};
	return call_with_context(
		[&]() { return context->handle_read(fd, buf, count); },
		call_original);
}

extern "C" FILE *fopen(const char *pathname, const char *flags)
{
	auto call_original = [&]() {
		return call_next_symbol(next_fopen_symbol, "fopen",
					(FILE *)nullptr, pathname, flags);
	};
	return call_with_context_debug(
		"fopen {} {}",
		[&]() { return context->handle_fopen(pathname, flags); },
		call_original, safe_ptr_str(pathname), safe_ptr_str(flags));
}
extern "C" FILE *fopen64(const char *pathname, const char *flags)
{
	auto call_original = [&]() {
		return call_next_symbol(next_fopen64_symbol, "fopen64",
					(FILE *)nullptr, pathname, flags);
	};
	return call_with_context_debug(
		"fopen64 {} {}",
		[&]() { return context->handle_fopen(pathname, flags); },
		call_original, safe_ptr_str(pathname), safe_ptr_str(flags));
}
extern "C" FILE *_IO_new_fopen(const char *pathname, const char *flags)
{
	auto call_original = [&]() {
		return call_next_symbol(next_fopen_symbol, "fopen",
					(FILE *)nullptr, pathname, flags);
	};
	return call_with_context_debug(
		"_IO_new_fopen {} {}",
		[&]() { return context->handle_fopen(pathname, flags); },
		call_original, safe_ptr_str(pathname), safe_ptr_str(flags));
}
#if __linux__
extern "C" long syscall(long sysno, ...)
{
	// glibc directly reads the arguments without considering
	// the underlying argument number. So did us
	va_list args;
	va_start(args, sysno);
	long arg1 = va_arg(args, long);
	long arg2 = va_arg(args, long);
	long arg3 = va_arg(args, long);
	long arg4 = va_arg(args, long);
	long arg5 = va_arg(args, long);
	long arg6 = va_arg(args, long);
	va_end(args);
	if (!initialize_ctx())
		return fallback_syscall(sysno, arg1, arg2, arg3, arg4, arg5,
					arg6);
	auto call_original = [&]() {
		return context->orig_syscall_fn(sysno, arg1, arg2, arg3, arg4,
						arg5, arg6);
	};
	if (sysno == __NR_bpf) {
		safe_spdlog_debug("SYS_BPF {} {} {} {} {} {}", arg1, arg2, arg3,
				  arg4, arg5, arg6);
		int cmd = (int)arg1;
		auto attr = (union bpf_attr *)(uintptr_t)arg2;
		auto size = (size_t)arg3;
		return handle_exceptions(
			[&]() {
				return context->handle_sysbpf(cmd, attr, size);
			},
			call_original);
	} else if (sysno == __NR_perf_event_open) {
		safe_spdlog_debug("SYS_PERF_EVENT_OPEN {} {} {} {} {} {}", arg1,
				  arg2, arg3, arg4, arg5, arg6);
		return handle_exceptions(
			[&]() {
				return context->handle_perfevent(
					(perf_event_attr *)(uintptr_t)arg1,
					(pid_t)arg2, (int)arg3, (int)arg4,
					(unsigned long)arg5);
			},
			call_original);
	} else if (sysno == __NR_ioctl) {
		safe_spdlog_debug("SYS_IOCTL {} {} {} {} {} {}", arg1, arg2,
				  arg3, arg4, arg5, arg6);
	} else if (sysno == __NR_dup3) {
		safe_spdlog_debug("SYS_DUP3 oldfd={} newfd={} flags={}", arg1,
				  arg2, arg3);
		return handle_exceptions(
			[&]() {
				return context->handle_dup3(
					(int)arg1, (int)arg2, (int)arg3);
			},
			call_original);
	} else if (sysno == __NR_memfd_create) {
		safe_spdlog_debug("SYS_MEMFD_CREATE name={} flags={}",
				  safe_ptr_str((const char *)arg1), arg2);
		return handle_exceptions(
			[&]() {
				return context->handle_memfd_create(
					(const char *)arg1, (int)arg2);
			},
			call_original);
	}
	return call_original();
}
#endif

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
extern "C" int bpftime_syscall_server__poll_gpu_ringbuf_map(
	int mapfd, void *ctx, void (*fn)(const void *, uint64_t, void *))
{
	if (!initialize_ctx()) {
		errno = ENOSYS;
		return -1;
	}
	return handle_exceptions(
		[&]() { return context->poll_gpu_ringbuf_map(mapfd, ctx, fn); },
		[&]() {
			errno = EIO;
			return -1;
		});
}
#endif
