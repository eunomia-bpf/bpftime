#ifndef _SYSCALL_CONTEXT_HPP
#define _SYSCALL_CONTEXT_HPP
#include "linux/perf_event.h"
#include <cstddef>
#include <dlfcn.h>
#include <sys/types.h>
#include <spdlog/spdlog.h>
#include <unordered_set>
class syscall_context {
	using syscall_fn = long (*)(long, ...);
	using close_fn = int (*)(int);
	using mmap64_fn = void *(*)(void *, size_t, int, int, int, off64_t);
	using ioctl_fn = int (*)(int fd, unsigned long req, int);
	using epoll_craete1_fn = int (*)(int);
	using epoll_ctl_fn = int (*)(int, int, int, struct epoll_event *);
	using epoll_wait_fn = int (*)(int, struct epoll_event *, int, int);
	using munmap_fn = int (*)(void *, size_t);
	close_fn orig_close_fn = nullptr;
	mmap64_fn orig_mmap64_fn = nullptr;
	ioctl_fn orig_ioctl_fn = nullptr;
	epoll_craete1_fn orig_epoll_create1_fn = nullptr;
	epoll_ctl_fn orig_epoll_ctl_fn = nullptr;
	epoll_wait_fn orig_epoll_wait_fn = nullptr;
	munmap_fn orig_munmap_fn = nullptr;
	std::unordered_set<uintptr_t> mocked_mmap_values;
	void init_original_functions()
	{
		orig_epoll_wait_fn =
			(epoll_wait_fn)dlsym(RTLD_NEXT, "epoll_wait");
		orig_epoll_ctl_fn = (epoll_ctl_fn)dlsym(RTLD_NEXT, "epoll_ctl");
		orig_epoll_create1_fn =
			(epoll_craete1_fn)dlsym(RTLD_NEXT, "epoll_create1");
		orig_ioctl_fn = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
		orig_syscall_fn = (syscall_fn)dlsym(RTLD_NEXT, "syscall");
		orig_mmap64_fn = (mmap64_fn)dlsym(RTLD_NEXT, "mmap");
		orig_close_fn = (close_fn)dlsym(RTLD_NEXT, "close");
		orig_munmap_fn = (munmap_fn)dlsym(RTLD_NEXT, "munmap");
	}

    public:
	syscall_context()
	{
		init_original_functions();
		spdlog::info("manager constructed");
	}
	syscall_fn orig_syscall_fn = nullptr;

	// handle syscall
	int handle_close(int);
	long handle_sysbpf(int cmd, union bpf_attr *attr, size_t size);
	int handle_perfevent(perf_event_attr *attr, pid_t pid, int cpu,
			     int group_fd, unsigned long flags);
	void *handle_mmap64(void *addr, size_t length, int prot, int flags,
			    int fd, off_t offset);
	int handle_ioctl(int fd, unsigned long req, int data);
	int handle_epoll_create1(int);
	int handle_epoll_ctl(int epfd, int op, int fd, epoll_event *evt);
	int handle_epoll_wait(int epfd, epoll_event *evt, int maxevents,
			      int timeout);
	int handle_munmap(void *addr, size_t size);
};

#endif
