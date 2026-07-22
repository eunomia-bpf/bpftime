/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _SYSCALL_CONTEXT_HPP
#define _SYSCALL_CONTEXT_HPP
#include <set>
#include <unordered_map>
#if defined(__linux__)
#include "linux/perf_event.h"
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/types.h>
#include <spdlog/spdlog.h>
#include <unordered_set>
#include <pthread.h>
#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
// #include "pos/include/common.h"
// #include "pos/include/utils/command_caller.h"
// #include "pos/cli/cli.h"
#if __APPLE__
using namespace bpftime_epoll;
#endif

constexpr const int MOCKED_UPROBE_TYPE_VALUE = 9;
constexpr const int MOCKED_URETPROBE_BIT = 0;

constexpr const int MOCKED_KPROBE_TYPE_VALUE = 8;
constexpr const int MOCKED_KRETPROBE_BIT = 0;

struct mocked_file_provider {
	/**
	 * @brief Next available byte
	 *
	 */
	int cursor = 0;
	std::string buf;
	pthread_spinlock_t access_lock;
	FILE *replacement_file = nullptr;
	mocked_file_provider(std::string buf) : buf(buf)
	{
		pthread_spin_init(&access_lock, 0);
	}
	virtual ~mocked_file_provider()
	{
		pthread_spin_destroy(&access_lock);
	}
	mocked_file_provider(const mocked_file_provider &) = delete;
	mocked_file_provider &operator=(const mocked_file_provider &) = delete;

	mocked_file_provider(mocked_file_provider &&) = default;
	mocked_file_provider &operator=(mocked_file_provider &&) = default;
};

class syscall_context {
	using syscall_fn = long (*)(long, ...);
	using close_fn = int (*)(int);
	using mmap64_fn = void *(*)(void *, size_t, int, int, int, off64_t);
	using mmap_fn = void *(*)(void *, size_t, int, int, int, off_t);
	using ioctl_fn = int (*)(int fd, unsigned long req, unsigned long);
	using epoll_craete1_fn = int (*)(int);
	using epoll_ctl_fn = int (*)(int, int, int, struct epoll_event *);
	using epoll_wait_fn = int (*)(int, struct epoll_event *, int, int);
	using munmap_fn = int (*)(void *, size_t);
	using openat_fn = int (*)(int, const char *, int, ...);
	using open_fn = int (*)(const char *, int, ...);
	using read_fn = ssize_t (*)(int fd, void *buf, size_t count);
	using fopen_fn = FILE *(*)(const char *, const char *);

	close_fn orig_close_fn = nullptr;
	mmap64_fn orig_mmap64_fn = nullptr;
	ioctl_fn orig_ioctl_fn = nullptr;
	epoll_craete1_fn orig_epoll_create1_fn = nullptr;
	epoll_ctl_fn orig_epoll_ctl_fn = nullptr;
	epoll_wait_fn orig_epoll_wait_fn = nullptr;
	munmap_fn orig_munmap_fn = nullptr;
	openat_fn orig_openat_fn = nullptr;
	open_fn orig_open_fn = nullptr;
	mmap_fn orig_mmap_fn = nullptr;
	read_fn orig_read_fn = nullptr;
	fopen_fn orig_fopen_fn = nullptr;

	std::unordered_set<uintptr_t> mocked_mmap_values;
	pthread_spinlock_t mocked_file_lock;
	std::unordered_map<int, std::unique_ptr<mocked_file_provider>>
		mocked_files;
	template <typename Fn> static Fn resolve_original(const char *name)
	{
		auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
		if (fn == nullptr)
			throw std::runtime_error(
				std::string("Unable to resolve ") + name);
		return fn;
	}
	void init_original_functions()
	{
		orig_epoll_wait_fn =
			resolve_original<epoll_wait_fn>("epoll_wait");
		orig_epoll_ctl_fn = resolve_original<epoll_ctl_fn>("epoll_ctl");
		orig_epoll_create1_fn =
			resolve_original<epoll_craete1_fn>("epoll_create1");
		orig_ioctl_fn = resolve_original<ioctl_fn>("ioctl");
		orig_syscall_fn = resolve_original<syscall_fn>("syscall");
		orig_close_fn = resolve_original<close_fn>("close");
		orig_munmap_fn = resolve_original<munmap_fn>("munmap");
		orig_mmap64_fn = orig_mmap_fn =
			resolve_original<mmap_fn>("mmap");
		orig_openat_fn = resolve_original<openat_fn>("openat");
		orig_open_fn = resolve_original<open_fn>("open");
		orig_read_fn = resolve_original<read_fn>("read");
		orig_fopen_fn =
			reinterpret_cast<fopen_fn>(dlsym(RTLD_NEXT, "fopen"));
		if (orig_fopen_fn == nullptr)
			orig_fopen_fn = resolve_original<fopen_fn>("fopen64");

		// To avoid polluting other child processes,
		// unset the LD_PRELOAD env var after syscall context being
		// initialized
		unsetenv("LD_PRELOAD");
		// Keep this log allocation-free. spdlog/fmt may throw in
		// extremely early init paths (and will print "[*** LOG ERROR
		// ***]").
		SPDLOG_DEBUG("Resolved original libc function pointers");
	}

	int create_kernel_bpf_map(int fd, int bpftime_map_type);
	int create_kernel_bpf_prog_in_userspace(int cmd, union bpf_attr *attr,
						size_t size);
	// try loading the bpf syscall helpers.
	// if the syscall original function is not prepared, it will cause a
	// segfault.
	bool try_startup() noexcept;

	// enable userspace eBPF runing with kernel eBPF.
	bool run_with_kernel = false;
	// allow programs to by pass the verifier
	// some extensions are not supported by the verifier, so we need to
	// by pass the verifier to make it work.
	std::string by_pass_kernel_verifier_pattern;

	void load_config_from_env();

    public:
	// enable mock the syscall behavior in userspace
	std::atomic<bool> enable_mock{ true };
	// Initializing CUDA
	std::atomic<bool> initializing_cuda{ false };
	// Whether enable mock after syscall server has been initialized
	std::atomic<bool> enable_mock_after_initialized{ true };
	void disable_mocking() noexcept
	{
		enable_mock.store(false, std::memory_order_relaxed);
		enable_mock_after_initialized.store(false,
						    std::memory_order_relaxed);
	}
	syscall_context();
	virtual ~syscall_context()
	{
		pthread_spin_destroy(&this->mocked_file_lock);
	}
	syscall_fn orig_syscall_fn = nullptr;

	// handle syscall
	int handle_close(int);
	long handle_sysbpf(int cmd, union bpf_attr *attr, size_t size);
	int handle_perfevent(perf_event_attr *attr, pid_t pid, int cpu,
			     int group_fd, unsigned long flags);
	void *handle_mmap64(void *addr, size_t length, int prot, int flags,
			    int fd, off64_t offset);
	void *handle_mmap(void *addr, size_t length, int prot, int flags,
			  int fd, off_t offset);

	int handle_ioctl(int fd, unsigned long req, unsigned long data);
	int handle_epoll_create1(int);
	int handle_epoll_ctl(int epfd, int op, int fd, epoll_event *evt);
	int handle_epoll_wait(int epfd, epoll_event *evt, int maxevents,
			      int timeout);
	int handle_munmap(void *addr, size_t size);
	int handle_openat(int fd, const char *file, int oflag,
			  unsigned short mode);
	int handle_open(const char *file, int oflag, unsigned short mode);
	ssize_t handle_read(int fd, void *buf, size_t count);
	FILE *handle_fopen(const char *pathname, const char *flags);
	int handle_dup3(int oldfd, int newfd, int flags);
	int handle_memfd_create(const char *name, int flags);

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
	void initialize_cuda();
	int poll_gpu_ringbuf_map(int mapfd, void *ctx,

				 void (*)(const void *, uint64_t, void *));
#endif
};

#endif
