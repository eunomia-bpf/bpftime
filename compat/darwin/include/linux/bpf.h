/* SPDX-License-Identifier: MIT
 *
 * Darwin compatibility shim for the Linux uapi <linux/bpf.h> constants.
 *
 * bpftime's cross-platform runtime core references the kernel BPF uapi map
 * types and update/stack flags. On Linux these come from the real
 * <linux/bpf.h>. macOS has no such header, so this shim provides the stable
 * uapi values with identical numeric meaning. It is only put on the include
 * path for Apple builds (see runtime/CMakeLists.txt); Linux continues to
 * include the genuine kernel header and never sees this file.
 *
 * Map types are declared as an unscoped enum (not macros) so they do NOT
 * collide with bpftime's own `enum class bpf_map_type` members: an enum
 * constant named BPF_MAP_TYPE_HASH is a different entity from
 * bpftime::bpf_map_type::BPF_MAP_TYPE_HASH, whereas a macro would corrupt
 * the latter by preprocessor substitution. Numeric values track the kernel
 * uapi ABI (include/uapi/linux/bpf.h), which also matches the declaration
 * order of bpftime's enum class bpf_map_type.
 *
 * Only the constants referenced by the cross-platform core are defined.
 * Linux-only facilities (seccomp, syscall tracing, perf-event array, libbpf
 * interop) stay behind __linux__ guards and are not mirrored here.
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
#define BPF_F_RDONLY (1U << 3)
#define BPF_F_WRONLY (1U << 4)

/* Stack-trace map flags. */
#define BPF_F_USER_STACK (1ULL << 8)
#define BPF_F_FAST_STACK_CMP (1ULL << 9)
#define BPF_F_REUSE_STACKID (1ULL << 10)
#define BPF_F_USER_BUILD_ID (1ULL << 11)
#define BPF_F_SKIP_FIELD_MASK 0xffULL

/*
 * enum bpf_map_type kernel uapi values. Unscoped enum constants coexist with
 * bpftime's enum class bpf_map_type without macro substitution. The legacy
 * BPF_MAP_TYPE_CGROUP_STORAGE alias shares value 19; no BPF_MAP_TYPE_MAX is
 * defined here because bpftime extends the type space well past the kernel
 * bound in its own enum class (see bpftime_shm.hpp), so a kernel-valued MAX
 * would be wrong for the cross-platform code.
 */
enum bpf_map_type_compat {
	BPF_MAP_TYPE_UNSPEC = 0,
	BPF_MAP_TYPE_HASH = 1,
	BPF_MAP_TYPE_ARRAY = 2,
	BPF_MAP_TYPE_PROG_ARRAY = 3,
	BPF_MAP_TYPE_PERF_EVENT_ARRAY = 4,
	BPF_MAP_TYPE_PERCPU_HASH = 5,
	BPF_MAP_TYPE_PERCPU_ARRAY = 6,
	BPF_MAP_TYPE_STACK_TRACE = 7,
	BPF_MAP_TYPE_CGROUP_ARRAY = 8,
	BPF_MAP_TYPE_LRU_HASH = 9,
	BPF_MAP_TYPE_LRU_PERCPU_HASH = 10,
	BPF_MAP_TYPE_LPM_TRIE = 11,
	BPF_MAP_TYPE_ARRAY_OF_MAPS = 12,
	BPF_MAP_TYPE_HASH_OF_MAPS = 13,
	BPF_MAP_TYPE_DEVMAP = 14,
	BPF_MAP_TYPE_SOCKMAP = 15,
	BPF_MAP_TYPE_CPUMAP = 16,
	BPF_MAP_TYPE_XSKMAP = 17,
	BPF_MAP_TYPE_SOCKHASH = 18,
	BPF_MAP_TYPE_CGROUP_STORAGE = 19,
	BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED = 19,
	BPF_MAP_TYPE_REUSEPORT_SOCKARRAY = 20,
	BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE = 21,
	BPF_MAP_TYPE_QUEUE = 22,
	BPF_MAP_TYPE_STACK = 23,
	BPF_MAP_TYPE_SK_STORAGE = 24,
	BPF_MAP_TYPE_DEVMAP_HASH = 25,
	BPF_MAP_TYPE_STRUCT_OPS = 26,
	BPF_MAP_TYPE_RINGBUF = 27,
	BPF_MAP_TYPE_INODE_STORAGE = 28,
	BPF_MAP_TYPE_TASK_STORAGE = 29,
	BPF_MAP_TYPE_BLOOM_FILTER = 30,
	BPF_MAP_TYPE_USER_RINGBUF = 31,
	BPF_MAP_TYPE_CGRP_STORAGE = 32,
};

#endif /* BPFTIME_COMPAT_DARWIN_LINUX_BPF_H */
