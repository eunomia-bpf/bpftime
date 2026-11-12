/* SPDX-License-Identifier: MIT
 *
 * Darwin compatibility shim for the Linux uapi <linux/bpf.h> constants.
 *
 * bpftime's cross-platform runtime core references the kernel BPF uapi enum
 * values (map types, update/stack flags) unqualified. On Linux these come from
 * the real <linux/bpf.h>. macOS has no such header, so this shim provides the
 * stable uapi values with identical numeric meaning. It is only put on the
 * include path for Apple builds (see runtime/CMakeLists.txt); Linux continues
 * to include the genuine kernel header and never sees this file.
 *
 * Only the constants actually referenced by the cross-platform core are
 * defined. Linux-only facilities (seccomp, syscall tracing, the perf-event
 * array, libbpf interop) stay behind __linux__ guards and are not mirrored.
 */
#ifndef BPFTIME_COMPAT_DARWIN_LINUX_BPF_H
#define BPFTIME_COMPAT_DARWIN_LINUX_BPF_H

#include <cstdint>

/* Map update operations (include/uapi/linux/bpf.h BPF_MAP_*). */
#define BPF_ANY 0
#define BPF_NOEXIST 1
#define BPF_EXIST 2
#define BPF_F_LOCK 4

/* Map creation flags. */
#define BPF_F_RDONLY (1U << 2)
#define BPF_F_WRONLY (1U << 3)

/* Stack-trace map flags. */
#define BPF_F_USER_STACK (1ULL << 8)
#define BPF_F_FAST_STACK_CMP (1ULL << 9)
#define BPF_F_REUSE_STACKID (1ULL << 10)
#define BPF_F_SKIP_FIELD_MASK 0xffULL
#define BPF_F_USER_BUILD_ID (-1LL)

/*
 * enum bpf_map_type values. The numeric sequence is the kernel uapi ABI and
 * matches the order of bpftime's own `enum class bpf_map_type` in
 * bpftime_shm.hpp, so casts between the two stay correct on every platform.
 */
#define BPF_MAP_TYPE_UNSPEC 0
#define BPF_MAP_TYPE_HASH 1
#define BPF_MAP_TYPE_ARRAY 2
#define BPF_MAP_TYPE_PROG_ARRAY 3
#define BPF_MAP_TYPE_PERF_EVENT_ARRAY 4
#define BPF_MAP_TYPE_PERCPU_HASH 5
#define BPF_MAP_TYPE_PERCPU_ARRAY 6
#define BPF_MAP_TYPE_STACK_TRACE 7
#define BPF_MAP_TYPE_CGROUP_ARRAY 8
#define BPF_MAP_TYPE_LRU_HASH 9
#define BPF_MAP_TYPE_LRU_PERCPU_HASH 10
#define BPF_MAP_TYPE_LPM_TRIE 11
#define BPF_MAP_TYPE_ARRAY_OF_MAPS 12
#define BPF_MAP_TYPE_HASH_OF_MAPS 13
#define BPF_MAP_TYPE_DEVMAP 14
#define BPF_MAP_TYPE_SOCKMAP 15
#define BPF_MAP_TYPE_CPUMAP 16
#define BPF_MAP_TYPE_XSKMAP 17
#define BPF_MAP_TYPE_SOCKHASH 18
#define BPF_MAP_TYPE_CGROUP_STORAGE 19
#define BPF_MAP_TYPE_REUSEPORT_SOCKARRAY 20
#define BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE 21
#define BPF_MAP_TYPE_QUEUE 22
#define BPF_MAP_TYPE_STACK 23
#define BPF_MAP_TYPE_SK_STORAGE 24
#define BPF_MAP_TYPE_DEVMAP_HASH 25
#define BPF_MAP_TYPE_STRUCT_OPS 26
#define BPF_MAP_TYPE_RINGBUF 27
#define BPF_MAP_TYPE_INODE_STORAGE 28
#define BPF_MAP_TYPE_TASK_STORAGE 29
#define BPF_MAP_TYPE_BLOOM_FILTER 30
#define BPF_MAP_TYPE_USER_RINGBUF 31
#define BPF_MAP_TYPE_CGRP_STORAGE 32
#define BPF_MAP_TYPE_MAX 33

#endif /* BPFTIME_COMPAT_DARWIN_LINUX_BPF_H */
