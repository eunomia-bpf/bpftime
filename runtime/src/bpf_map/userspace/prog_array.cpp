/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#if __linux__
#include "linux/bpf.h"
#include <asm/unistd.h>
#include <unistd.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include "bpftime_shm.hpp"
#include <bpf_map/userspace/prog_array.hpp>
#include <cerrno>
#include <spdlog/spdlog.h>
#include <stdexcept>

#ifndef offsetofend
#define offsetofend(TYPE, FIELD)                                               \
	(offsetof(TYPE, FIELD) + sizeof(((TYPE *)0)->FIELD))
#endif

#if __linux__

// Issue the bpf syscall directly instead of going through libc's syscall()
// symbol. On x86_64/aarch64 the bpftime syscall server interposes that
// symbol, so a plain ::syscall() would re-enter bpftime emulation instead of
// reaching the kernel. The raw instruction takes no dynamic-loader lock,
// does no allocation and needs no dlopen/dlsym. Encoded userspace entries
// return before this kernel-only path, keeping their lookup syscall-free.
static long raw_bpf_syscall(long cmd, union bpf_attr *attr,
			    unsigned long size)
{
#if defined(__x86_64__) || defined(__aarch64__)
#if defined(__x86_64__)
	long ret;
	__asm__ volatile("syscall"
			 : "=a"(ret)
			 : "a"((long)__NR_bpf), "D"(cmd), "S"(attr),
			   "d"((long)size)
			 : "rcx", "r11", "memory", "cc");
#else
	register long x8 __asm__("x8") = __NR_bpf;
	register long x0 __asm__("x0") = cmd;
	register long x1 __asm__("x1") = reinterpret_cast<long>(attr);
	register long x2 __asm__("x2") = (long)size;
	__asm__ volatile("svc #0"
			 : "=r"(x0)
			 : "r"(x8), "0"(x0), "r"(x1), "r"(x2)
			 : "memory", "cc");
	long ret = x0;
#endif
	if (ret < 0) {
		errno = (int)-ret;
		return -1;
	}
	return ret;
#else
	// Other Linux architectures do not build the syscall() interposer,
	// so the libc wrapper reaches the kernel directly.
	return ::syscall(__NR_bpf, cmd, attr, size);
#endif
}

static int my_bpf_obj_get_info_by_fd(int bpf_fd, void *info, __u32 *info_len)
{
	const size_t attr_sz = offsetofend(union bpf_attr, info);
	union bpf_attr attr;
	int err;

	memset(&attr, 0, attr_sz);
	attr.info.bpf_fd = bpf_fd;
	attr.info.info_len = *info_len;
	attr.info.info = (uintptr_t)info;

	err = raw_bpf_syscall(BPF_OBJ_GET_INFO_BY_FD, &attr, attr_sz);
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
		fd = raw_bpf_syscall(cmd, attr, size);
		// The wrapper keeps libc conventions (-1 + errno), so the
		// retry inspects errno rather than a raw -EAGAIN return.
	} while (fd < 0 && errno == EAGAIN && --attempts > 0);
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
namespace
{
constexpr int32_t INVALID_ENTRY = -1;

constexpr bool is_userspace_prog_fd_entry(int32_t value)
{
	return value < INVALID_ENTRY;
}

constexpr int32_t encode_userspace_prog_fd(int fd)
{
	return -fd - 2;
}

constexpr int decode_userspace_prog_fd(int32_t value)
{
	return -value - 2;
}
} // namespace

// The runtime is compiled with -ftls-model=initial-exec, so this access never
// reaches __tls_get_addr while libc holds an allocator lock.
static thread_local int32_t current_thread_lookup_val = 0;

prog_array_map_impl::prog_array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	: data(max_entries, INVALID_ENTRY, memory.get_segment_manager())
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
	int32_t value = data[k];
	if (value == INVALID_ENTRY) {
		errno = ENOENT;
		return nullptr;
	}
	if (is_userspace_prog_fd_entry(value)) {
		int fd = decode_userspace_prog_fd(value);
		if (!bpftime_is_prog_fd(fd)) {
			errno = ENOENT;
			return nullptr;
		}
		current_thread_lookup_val = fd;
		return &current_thread_lookup_val;
	}
	int fd = my_bpf_prog_get_fd_by_id(value);
	if (fd < 0) {
		SPDLOG_ERROR("Unable to retrieve prog fd of id {}", value);
		errno = ENOENT;
		return nullptr;
	}
	SPDLOG_DEBUG("prog array: fd of prog id {} is {}", value, fd);
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
	if (bpftime_is_prog_fd(v)) {
		data[k] = encode_userspace_prog_fd(v);
		SPDLOG_DEBUG("prog array: update slot {} to bpftime prog fd {}",
			     k, v);
		return 0;
	}
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
	data[k] = INVALID_ENTRY;
	return 0;
}

int prog_array_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Kernel array_map_get_next_key semantics: an absent or out-of-range
	// key restarts at index 0 instead of failing, and the last index is
	// the only one that reports -ENOENT. Array iteration is dense, so
	// empty slots are still visited.
	uint32_t index = key != nullptr ? *(const uint32_t *)key : UINT32_MAX;
	uint32_t *out = (uint32_t *)next_key;

	if (index >= data.size()) {
		*out = 0;
		return 0;
	}
	if (index == data.size() - 1) {
		errno = ENOENT;
		return -1;
	}
	*out = index + 1;
	return 0;
}

} // namespace bpftime
