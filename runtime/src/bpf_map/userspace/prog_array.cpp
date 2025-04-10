/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#if __linux__
#include "bpf/bpf.h"
#include "linux/bpf.h"
#include <bpf/libbpf.h>
#include <gnu/lib-names.h>
#include <asm/unistd.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include "spdlog/spdlog.h"
#include <bpf_map/userspace/prog_array.hpp>
#include <cerrno>
#include <spdlog/spdlog.h>
#include <dlfcn.h>
#include <stdexcept>

#ifndef offsetofend
#define offsetofend(TYPE, FIELD)                                               \
	(offsetof(TYPE, FIELD) + sizeof(((TYPE *)0)->FIELD))
#endif


#if __linux__

// syscall() function was hooked by syscall server, direct call to it will lead
// to a result provided by bpftime. So if we want to get things from kernel, we
// must manually execute `syscall` from libc
static void *libc_handle = dlopen(LIBC_SO, RTLD_LAZY);
static auto libc_syscall =
	reinterpret_cast<decltype(&::syscall)>(dlsym(libc_handle, "syscall"));

static int my_bpf_obj_get_info_by_fd(int bpf_fd, void *info, __u32 *info_len)
{
	const size_t attr_sz = offsetofend(union bpf_attr, info);
	union bpf_attr attr;
	int err;

	memset(&attr, 0, attr_sz);
	attr.info.bpf_fd = bpf_fd;
	attr.info.info_len = *info_len;
	attr.info.info = (uintptr_t)info;

	err = libc_syscall(__NR_bpf, BPF_OBJ_GET_INFO_BY_FD, &attr, attr_sz);
	if (!err)
		*info_len = attr.info.info_len;
	return err;
}

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

int my_bpf_prog_get_fd_by_id(__u32 id)
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
static thread_local uint32_t current_thread_lookup_val = 0;

prog_array_map_impl::prog_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	// Default value of fd map is -1
	: data(max_entries, -1, memory.get_segment_manager())
{
	if (key_size != 4 || value_size != 4) {
		throw std::runtime_error(
			"Key size and value size of prog_array must be 4");
	}
}

#if __linux__

void *prog_array_map_impl::elem_lookup(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return nullptr;
	}
	int fd = my_bpf_prog_get_fd_by_id(data[k]);
	if (fd < 0) {
		SPDLOG_ERROR("Unable to retrive prog fd of id {}", data[k]);
	}
	SPDLOG_DEBUG("prog array: fd of prog id {} is {}", data[k], fd);
	current_thread_lookup_val = fd;
	return &current_thread_lookup_val;
}

long prog_array_map_impl::elem_update(const void *key, const void *value,
				      uint64_t flags)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	int32_t v = *(int32_t *)value;
	struct bpf_prog_info info = {};
	uint32_t len = sizeof(info);
	int err = my_bpf_obj_get_info_by_fd(v, &info, &len);
	if (err < 0) {
		SPDLOG_ERROR(
			"Unable to get info of fd {} when adding into prog array: {}",
			v, err);
		return -1;
	}
	data[k] = info.id;
	SPDLOG_DEBUG("prog array: update slot {} to prog fd {}, id {}", k, v,
		     info.id);
	return 0;
}

#endif

long prog_array_map_impl::elem_delete(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	data[k] = -1;
	return 0;
}

int prog_array_map_impl::map_get_next_key(const void *key, void *next_key)
{
	int32_t *out = (int32_t *)next_key;
	if (key == nullptr) {
		*out = 0;
		return 0;
	}
	int32_t k = *(int32_t *)key;
	// The last key
	if ((size_t)(k + 1) == data.size()) {
		errno = ENOENT;
		return -1;
	}
	if (k < 0 || (size_t)k >= data.size()) {
		errno = EINVAL;
		return -1;
	}
	*out = k + 1;
	return 0;
}

} // namespace bpftime
