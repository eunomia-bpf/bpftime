/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "syscall_context.hpp"
#include "bpftime_logger.hpp"
#if defined(__aarch64__)
#include <asm-generic/unistd.h>
#else
#include <asm/unistd_64.h>
#endif
#include <boost/interprocess/exceptions.hpp>
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#if __linux__
#include "linux/bpf.h"
#include <asm-generic/errno-base.h>
#endif
#include <cstdlib>
#include <cstring>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <spdlog/cfg/env.h>
#include <cstdarg>
#include <sys/mman.h>
#include <utility>

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
	try {
		if (spdlog::default_logger_raw()) {
			spdlog::debug(fmt, std::forward<Args>(args)...);
		}
	} catch (...) {
	}
}

template <typename... Args>
inline void safe_spdlog_error(spdlog::format_string_t<Args...> fmt,
			      Args &&...args) noexcept
{
	try {
		if (spdlog::default_logger_raw()) {
			spdlog::error(fmt, std::forward<Args>(args)...);
		}
	} catch (...) {
	}
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
// 0 = uninitialized, 1 = in progress, 2 = ready, 3 = failed
static int ctx_initialized = 0;
static __thread int tls_initializing = 0;
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
		bpftime::bpftime_set_quiet_logger();
		try {
			new (&context.ctx) syscall_context;
		} catch (...) {
			tls_initializing = 0;
			__atomic_store_n(&ctx_initialized, 3, __ATOMIC_RELEASE);
			return false;
		}
		tls_initializing = 0;
		__atomic_store_n(&ctx_initialized, 2, __ATOMIC_RELEASE);
	} else {
		// Another thread is initializing bpftime. Do not block the
		// host; this call can safely use the original operation.
		return __atomic_load_n(&ctx_initialized, __ATOMIC_ACQUIRE) == 2;
	}
	return true;
}

#if defined(BPFTIME_PRELOAD_TEST_HOOKS)
extern "C" void bpftime_test_enable_syscall_mocking()
{
	if (!initialize_ctx())
		return;
	context->enable_mock.store(true, std::memory_order_relaxed);
	context->enable_mock_after_initialized.store(true,
						     std::memory_order_relaxed);
}
#endif

template <typename Fn, typename Result, typename... Args>
Result call_original(const char *name, Result failure, int caller_errno,
		     Args... args) noexcept
{
	auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
	if (fn == nullptr) {
		errno = ENOSYS;
		return failure;
	}
	errno = caller_errno;
	Result result = fn(args...);
	if (result != failure)
		errno = caller_errno;
	return result;
}

template <typename Result, typename Failure>
Result preserve_errno_on_success(Result result, Failure failure,
				 int caller_errno) noexcept
{
	if (result != static_cast<Result>(failure))
		errno = caller_errno;
	return result;
}

template <typename F, typename Fallback>
auto handle_exceptions(F &&f, Fallback &&fallback) noexcept -> decltype(f())
{
	try {
		return f();
	} catch (const boost::interprocess::bad_alloc &e) {
		safe_spdlog_error("Boost interprocess bad_alloc: {}", e.what());
		safe_spdlog_error(
			"Consider increasing the shared memory size by "
			"setting BPFTIME_SHM_MEMORY_MB.");
	} catch (const std::exception &e) {
		safe_spdlog_error("bpftime syscall interposition failed: {}",
				  e.what());
	} catch (...) {
		safe_spdlog_error(
			"bpftime syscall interposition failed with an unknown error");
	}
	context->disable_mocking();
	return fallback();
}

template <typename Failure, typename F, typename Fallback>
auto guard_call(Failure failure, int caller_errno, F &&f,
		Fallback &&fallback) noexcept -> decltype(f())
{
	return preserve_errno_on_success(
		handle_exceptions(std::forward<F>(f),
				  std::forward<Fallback>(fallback)),
		failure, caller_errno);
}

template <typename Failure, typename F, typename Fallback>
auto interpose(Failure failure, int caller_errno, F &&f,
	       Fallback &&fallback) noexcept -> decltype(f())
{
	if (!initialize_ctx())
		return fallback();
	return guard_call(failure, caller_errno, std::forward<F>(f),
			  std::forward<Fallback>(fallback));
}

extern "C" int epoll_wait(int epfd, epoll_event *evt, int maxevents,
			  int timeout)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int, epoll_event *, int, int);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("epoll_wait {}", epfd);
			return context->handle_epoll_wait(epfd, evt, maxevents,
							  timeout);
		},
		[&]() {
			return call_original<fn_t>("epoll_wait", -1,
						   caller_errno, epfd, evt,
						   maxevents, timeout);
		});
}

extern "C" int epoll_ctl(int epfd, int op, int fd, epoll_event *evt)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int, int, int, epoll_event *);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("epoll_ctl {} {} {} {}", epfd, op, fd,
					  (uintptr_t)evt);
			return context->handle_epoll_ctl(epfd, op, fd, evt);
		},
		[&]() {
			return call_original<fn_t>("epoll_ctl", -1,
						   caller_errno, epfd, op, fd,
						   evt);
		});
}

extern "C" int epoll_create1(int flags)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("epoll_create1 {}", flags);
			return context->handle_epoll_create1(flags);
		},
		[&]() {
			return call_original<fn_t>("epoll_create1", -1,
						   caller_errno, flags);
		});
}

extern "C" int ioctl(int fd, unsigned long req, ...)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int, unsigned long, ...);
	va_list args;
	va_start(args, req);
	unsigned long arg3 = va_arg(args, long);
	va_end(args);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("ioctl {} {} {}", fd, req, arg3);
			return context->handle_ioctl(fd, req, arg3);
		},
		[&]() {
			return call_original<fn_t>("ioctl", -1, caller_errno,
						   fd, req, arg3);
		});
}

extern "C" void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
			off64_t offset)
{
	const int caller_errno = errno;
	using fn_t = void *(*)(void *, size_t, int, int, int, off64_t);
	return interpose(
		MAP_FAILED, caller_errno,
		[&]() {
			safe_spdlog_debug("mmap64 {:x}", (uintptr_t)addr);
			return context->handle_mmap64(addr, length, prot, flags,
						      fd, offset);
		},
		[&]() {
			return call_original<fn_t>("mmap64", MAP_FAILED,
						   caller_errno, addr, length,
						   prot, flags, fd, offset);
		});
}

extern "C" void *mmap(void *addr, size_t length, int prot, int flags, int fd,
		      off_t offset)
{
	const int caller_errno = errno;
	using fn_t = void *(*)(void *, size_t, int, int, int, off_t);
	return interpose(
		MAP_FAILED, caller_errno,
		[&]() {
			safe_spdlog_debug("mmap {:x}", (uintptr_t)addr);
			return context->handle_mmap(addr, length, prot, flags,
						    fd, offset);
		},
		[&]() {
			return call_original<fn_t>("mmap", MAP_FAILED,
						   caller_errno, addr, length,
						   prot, flags, fd, offset);
		});
}

extern "C" int munmap(void *addr, size_t size)
{
	const int caller_errno = errno;
	using fn_t = int (*)(void *, size_t);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("munmap {:x} {}", (uintptr_t)addr,
					  size);
			return context->handle_munmap(addr, size);
		},
		[&]() {
			return call_original<fn_t>("munmap", -1, caller_errno,
						   addr, size);
		});
}

extern "C" int close(int fd)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int);
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("Closing fd {}", fd);
			return context->handle_close(fd);
		},
		[&]() {
			return call_original<fn_t>("close", -1, caller_errno,
						   fd);
		});
}

extern "C" int openat(int fd, const char *file, int oflag, ...)
{
	const int caller_errno = errno;
	using fn_t = int (*)(int, const char *, int, ...);
	unsigned short mode = 0;
	bool needs_mode = (oflag & O_CREAT) != 0;
#ifdef O_TMPFILE
	needs_mode = needs_mode || (oflag & O_TMPFILE) == O_TMPFILE;
#endif
	va_list args;
	if (needs_mode) {
		va_start(args, oflag);
		mode = static_cast<unsigned short>(va_arg(args, int));
		va_end(args);
	}
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("openat {} {} {} {}", fd,
					  safe_ptr_str(file), oflag, mode);
			return context->handle_openat(fd, file, oflag, mode);
		},
		[&]() {
			return call_original<fn_t>("openat", -1, caller_errno,
						   fd, file, oflag, mode);
		});
}
extern "C" int open(const char *file, int oflag, ...)
{
	const int caller_errno = errno;
	using fn_t = int (*)(const char *, int, ...);
	unsigned short mode = 0;
	bool needs_mode = (oflag & O_CREAT) != 0;
#ifdef O_TMPFILE
	needs_mode = needs_mode || (oflag & O_TMPFILE) == O_TMPFILE;
#endif
	va_list args;
	if (needs_mode) {
		va_start(args, oflag);
		mode = static_cast<unsigned short>(va_arg(args, int));
		va_end(args);
	}
	return interpose(
		-1, caller_errno,
		[&]() {
			safe_spdlog_debug("open {} {} {}", safe_ptr_str(file),
					  oflag, mode);
			return context->handle_open(file, oflag, mode);
		},
		[&]() {
			return call_original<fn_t>("open", -1, caller_errno,
						   file, oflag, mode);
		});
}
extern "C" ssize_t read(int fd, void *buf, size_t count)
{
	const int caller_errno = errno;
	using fn_t = ssize_t (*)(int, void *, size_t);
	return interpose(
		static_cast<ssize_t>(-1), caller_errno,
		[&]() { return context->handle_read(fd, buf, count); },
		[&]() {
			return call_original<fn_t>("read",
						   static_cast<ssize_t>(-1),
						   caller_errno, fd, buf,
						   count);
		});
}

extern "C" FILE *fopen(const char *pathname, const char *flags)
{
	const int caller_errno = errno;
	using fn_t = FILE *(*)(const char *, const char *);
	return interpose(
		static_cast<FILE *>(nullptr), caller_errno,
		[&]() {
			safe_spdlog_debug("fopen {} {}", safe_ptr_str(pathname),
					  safe_ptr_str(flags));
			return context->handle_fopen(pathname, flags);
		},
		[&]() {
			return call_original<fn_t>("fopen",
						   static_cast<FILE *>(nullptr),
						   caller_errno, pathname,
						   flags);
		});
}
extern "C" FILE *fopen64(const char *pathname, const char *flags)
{
	const int caller_errno = errno;
	using fn_t = FILE *(*)(const char *, const char *);
	return interpose(
		static_cast<FILE *>(nullptr), caller_errno,
		[&]() {
			safe_spdlog_debug("fopen64 {} {}",
					  safe_ptr_str(pathname),
					  safe_ptr_str(flags));
			return context->handle_fopen(pathname, flags);
		},
		[&]() {
			return call_original<fn_t>("fopen64",
						   static_cast<FILE *>(nullptr),
						   caller_errno, pathname,
						   flags);
		});
}
extern "C" FILE *_IO_new_fopen(const char *pathname, const char *flags)
{
	const int caller_errno = errno;
	using fn_t = FILE *(*)(const char *, const char *);
	return interpose(
		static_cast<FILE *>(nullptr), caller_errno,
		[&]() {
			safe_spdlog_debug("_IO_new_fopen {} {}",
					  safe_ptr_str(pathname),
					  safe_ptr_str(flags));
			return context->handle_fopen(pathname, flags);
		},
		[&]() {
			return call_original<fn_t>("fopen",
						   static_cast<FILE *>(nullptr),
						   caller_errno, pathname,
						   flags);
		});
}
#if __linux__
extern "C" long syscall(long sysno, ...)
{
	const int caller_errno = errno;
	using fn_t = long (*)(long, ...);
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
		return call_original<fn_t>("syscall", -1L, caller_errno, sysno,
					   arg1, arg2, arg3, arg4, arg5, arg6);
	auto fallback = [&]() {
		return call_original<fn_t>("syscall", -1L, caller_errno, sysno,
					   arg1, arg2, arg3, arg4, arg5, arg6);
	};
	if (sysno == __NR_bpf) {
		safe_spdlog_debug("SYS_BPF {} {} {} {} {} {}", arg1, arg2, arg3,
				  arg4, arg5, arg6);
		int cmd = (int)arg1;
		auto attr = (union bpf_attr *)(uintptr_t)arg2;
		auto size = (size_t)arg3;
		return guard_call(
			-1L, caller_errno,
			[&]() {
				return context->handle_sysbpf(cmd, attr, size);
			},
			fallback);
	} else if (sysno == __NR_perf_event_open) {
		safe_spdlog_debug("SYS_PERF_EVENT_OPEN {} {} {} {} {} {}", arg1,
				  arg2, arg3, arg4, arg5, arg6);
		return guard_call(
			-1L, caller_errno,
			[&]() {
				return context->handle_perfevent(
					(perf_event_attr *)(uintptr_t)arg1,
					(pid_t)arg2, (int)arg3, (int)arg4,
					(unsigned long)arg5);
			},
			fallback);
	} else if (sysno == __NR_ioctl) {
		safe_spdlog_debug("SYS_IOCTL {} {} {} {} {} {}", arg1, arg2,
				  arg3, arg4, arg5, arg6);
	} else if (sysno == __NR_dup3) {
		safe_spdlog_debug("SYS_DUP3 oldfd={} newfd={} flags={}", arg1,
				  arg2, arg3);
		return guard_call(
			-1L, caller_errno,
			[&]() {
				return context->handle_dup3(
					(int)arg1, (int)arg2, (int)arg3);
			},
			fallback);
	} else if (sysno == __NR_memfd_create) {
		safe_spdlog_debug("SYS_MEMFD_CREATE name={} flags={}",
				  safe_ptr_str((const char *)arg1), arg2);
		return guard_call(
			-1L, caller_errno,
			[&]() {
				return context->handle_memfd_create(
					(const char *)arg1, (int)arg2);
			},
			fallback);
	}
	errno = caller_errno;
	return preserve_errno_on_success(
		context->orig_syscall_fn(sysno, arg1, arg2, arg3, arg4, arg5,
					 arg6),
		-1L, caller_errno);
}
#endif

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
extern "C" int bpftime_syscall_server__poll_gpu_ringbuf_map(
	int mapfd, void *ctx, void (*fn)(const void *, uint64_t, void *))
{
	if (!initialize_ctx()) {
		errno = EIO;
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
