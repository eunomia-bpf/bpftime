#include "syscall_context.hpp"
#include "bpftime_shm.hpp"
#include "linux/bpf.h"
#include <asm-generic/errno-base.h>
#include <cstdlib>
#include <cstring>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <spdlog/cfg/env.h>
#include <cstdarg>
using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_SERVER;

// global context for bpf syscall server
static syscall_context context;


extern "C" int epoll_wait(int epfd, epoll_event *evt, int maxevents,
			  int timeout)
{
	spdlog::debug("epoll_wait {}", epfd);
	return context.handle_epoll_wait(epfd, evt, maxevents, timeout);
}

extern "C" int epoll_ctl(int epfd, int op, int fd, epoll_event *evt)
{
	spdlog::debug("epoll_ctl {} {} {} {}", epfd, op, fd, (uintptr_t)evt);
	return context.handle_epoll_ctl(epfd, op, fd, evt);
}

extern "C" int epoll_create1(int flags)
{
	spdlog::debug("epoll_create1 {}", flags);
	return context.handle_epoll_create1(flags);
}

extern "C" int ioctl(int fd, unsigned long req, int data)
{
	spdlog::debug("ioctl {} {} {}", fd, req, data);
	return context.handle_ioctl(fd, req, data);
}

extern "C" void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
			off64_t offset)
{
	spdlog::debug("mmap64 {:x}", (uintptr_t)addr);
	return context.handle_mmap64(addr, length, prot, flags, fd, offset);
}

extern "C" int close(int fd)
{
	spdlog::debug("Closing {}", fd);
	return context.handle_close(fd);
}

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
	if (sysno == __NR_bpf) {
		spdlog::debug("SYS_BPF {} {} {} {} {} {}", arg1, arg2, arg3,
			      arg4, arg5, arg6);
		int cmd = (int)arg1;
		auto attr = (union bpf_attr *)(uintptr_t)arg2;
		auto size = (size_t)arg3;
		return context.handle_sysbpf(cmd, attr, size);
	} else if (sysno == __NR_perf_event_open) {
		spdlog::debug("SYS_PERF_EVENT_OPEN {} {} {} {} {} {}", arg1,
			      arg2, arg3, arg4, arg5, arg6);
		return context.handle_perfevent(
			(perf_event_attr *)(uintptr_t)arg1, (pid_t)arg2,
			(int)arg3, (int)arg4, (unsigned long)arg5);
	} else if (sysno == __NR_ioctl) {
		spdlog::debug("SYS_IOCTL {} {} {} {} {} {}", arg1, arg2, arg3,
			      arg4, arg5, arg6);
	}
	return context.orig_syscall_fn(sysno, arg1, arg2, arg3, arg4, arg5,
				       arg6);
}
