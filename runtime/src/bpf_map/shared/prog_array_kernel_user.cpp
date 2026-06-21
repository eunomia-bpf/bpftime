/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <bpf_map/shared/prog_array_kernel_user.hpp>

#include "bpftime_shm.hpp"
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#if __linux__
#include "bpf/bpf.h"
#include <asm/unistd.h>
#include <bpf/libbpf.h>
#include <gnu/lib-names.h>
#include <linux/bpf.h>
#endif

#ifndef offsetofend
#define offsetofend(TYPE, FIELD)                                               \
	(offsetof(TYPE, FIELD) + sizeof(((TYPE *)0)->FIELD))
#endif

#if __linux__
static void *libc_handle = dlopen(LIBC_SO, RTLD_LAZY);
static auto libc_syscall =
	reinterpret_cast<decltype(&::syscall)>(dlsym(libc_handle, "syscall"));

__attribute__((__noinline__, noinline)) static long
my_bpf_syscall_fd(long cmd, union bpf_attr *attr, unsigned long size)
{
	int attempts = 5;
	long fd;
	do {
		fd = libc_syscall(__NR_bpf, cmd, attr, size);
	} while (fd < 0 && fd == -EAGAIN && --attempts > 0);
	return fd;
}

static int my_bpf_prog_get_fd_by_id(__u32 id)
{
	const size_t attr_sz = offsetofend(union bpf_attr, open_flags);
	union bpf_attr attr;
	int fd;

	memset(&attr, 0, attr_sz);
	attr.prog_id = id;

	fd = my_bpf_syscall_fd(BPF_PROG_GET_FD_BY_ID, &attr, attr_sz);
	return fd;
}
#endif

namespace bpftime
{

static thread_local int32_t current_thread_lookup_val = 0;

prog_array_kernel_user_impl::prog_array_kernel_user_impl(
	boost::interprocess::managed_shared_memory &memory, int km_id)
	: kernel_map_id(km_id)
{
	(void)memory;
}

prog_array_kernel_user_impl::~prog_array_kernel_user_impl()
{
	if (map_fd >= 0) {
		close(map_fd);
	}
}

bool prog_array_kernel_user_impl::init_map_fd()
{
#if __linux__
	if (map_fd >= 0) {
		return true;
	}
	map_fd = bpf_map_get_fd_by_id(kernel_map_id);
	if (map_fd < 0) {
		SPDLOG_ERROR("Failed to get fd for kernel prog array id {}",
			     kernel_map_id);
		return false;
	}
	bpf_map_info info = {};
	unsigned int info_len = sizeof(info);
	if (bpf_obj_get_info_by_fd(map_fd, &info, &info_len) < 0) {
		SPDLOG_ERROR("Failed to get info for kernel prog array id {}",
			     kernel_map_id);
		return false;
	}
	_max_entries = info.max_entries;
	return true;
#else
	return false;
#endif
}

void *prog_array_kernel_user_impl::elem_lookup(const void *key)
{
#if __linux__
	if (!init_map_fd()) {
		return nullptr;
	}
	int32_t k = *(int32_t *)key;
	if (k < 0 || static_cast<uint32_t>(k) >= _max_entries) {
		errno = EINVAL;
		return nullptr;
	}
	uint32_t prog_id = 0;
	if (bpf_map_lookup_elem(map_fd, key, &prog_id) < 0) {
		return nullptr;
	}
	int fd = my_bpf_prog_get_fd_by_id(prog_id);
	if (fd < 0) {
		SPDLOG_ERROR("Unable to retrieve prog fd of id {}", prog_id);
		errno = ENOENT;
		return nullptr;
	}
	current_thread_lookup_val = fd;
	return &current_thread_lookup_val;
#else
	return nullptr;
#endif
}

void *prog_array_kernel_user_impl::elem_lookup_userspace(const void *key)
{
#if __linux__
	if (!init_map_fd()) {
		return nullptr;
	}
	int32_t k = *(int32_t *)key;
	if (k < 0 || static_cast<uint32_t>(k) >= _max_entries) {
		errno = EINVAL;
		return nullptr;
	}
	uint32_t prog_id = 0;
	if (bpf_map_lookup_elem(map_fd, key, &prog_id) < 0) {
		return nullptr;
	}
	current_thread_lookup_val = static_cast<int32_t>(prog_id);
	return &current_thread_lookup_val;
#else
	return nullptr;
#endif
}

long prog_array_kernel_user_impl::elem_update(const void *key,
					      const void *value, uint64_t flags)
{
#if __linux__
	if (!init_map_fd()) {
		return -1;
	}
	return bpf_map_update_elem(map_fd, key, value, flags);
#else
	return -1;
#endif
}

long prog_array_kernel_user_impl::elem_delete(const void *key)
{
#if __linux__
	if (!init_map_fd()) {
		return -1;
	}
	return bpf_map_delete_elem(map_fd, key);
#else
	return -1;
#endif
}

int prog_array_kernel_user_impl::map_get_next_key(const void *key,
						  void *next_key)
{
#if __linux__
	if (!init_map_fd()) {
		return -1;
	}
	return bpf_map_get_next_key(map_fd, key, next_key);
#else
	return -1;
#endif
}

} // namespace bpftime
