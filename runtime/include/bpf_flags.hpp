/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * BPF flag constants for map operations and stack trace operations.
 * Provides BPF_ANY, BPF_NOEXIST, BPF_EXIST, and BPF_F_* definitions.
 */
#ifndef BPFTIME_BPF_FLAGS_HPP
#define BPFTIME_BPF_FLAGS_HPP

#ifdef BPFTIME_BUILD_WITH_LIBBPF
#include <bpf/bpf.h>
#else
// Define BPF flags locally when libbpf is not available
// These match the definitions from linux/bpf.h
// Note: If linux/bpf.h is included elsewhere, it should be included
// before this header to avoid enum redefinition errors.
#ifndef BPFTIME_BPF_FLAGS_DEFINED
#define BPFTIME_BPF_FLAGS_DEFINED
enum {
	BPF_ANY = 0, /* create new element or update existing */
	BPF_NOEXIST = 1, /* create new element if it didn't exist */
	BPF_EXIST = 2, /* update existing element */
};

// Map flags
enum {
	BPF_F_RDONLY = (1U << 3),
	BPF_F_WRONLY = (1U << 4),
};

// Stack trace flags
enum {
	BPF_F_SKIP_FIELD_MASK = 0xffULL,
	BPF_F_USER_STACK = (1ULL << 8),
	BPF_F_FAST_STACK_CMP = (1ULL << 9),
	BPF_F_REUSE_STACKID = (1ULL << 10),
	BPF_F_USER_BUILD_ID = (1ULL << 11),
};
#endif // BPFTIME_BPF_FLAGS_DEFINED
#endif // BPFTIME_BUILD_WITH_LIBBPF

#endif // BPFTIME_BPF_FLAGS_HPP
