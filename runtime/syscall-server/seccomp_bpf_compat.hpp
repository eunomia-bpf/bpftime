/* SPDX-License-Identifier: MIT
 *
 * Portable classic BPF definitions for seccomp user-notification filters.
 *
 * Some glibc/kernel-headers combinations keep struct sock_filter and the
 * BPF_STMT/BPF_JUMP initializer macros out of the userspace-visible
 * <linux/filter.h>. The classic BPF/seccomp filter ABI is stable, so provide
 * guarded fallbacks. When the libc headers export these names, the #ifndef /
 * type-absent checks skip the definitions and there is no conflict.
 */
#ifndef BPFTIME_SECCOMP_BPF_COMPAT_HPP
#define BPFTIME_SECCOMP_BPF_COMPAT_HPP

#if defined(__linux__)
#include <linux/filter.h>
#include <linux/types.h>

// Detect whether the libc filter header exposed the classic BPF structures.
// BPF_STMT is the reliable sentinel across glibc/musl/bionic variants.
#if !defined(BPF_STMT)

struct sock_filter {
	__u16 code;
	__u8 jt;
	__u8 jf;
	__u32 k;
};

struct sock_fprog {
	unsigned short len;
	struct sock_filter *filter;
};

#define BPF_STMT(code, k)                                                     \
	{                                                                     \
		(unsigned short)(code), 0, 0, k                                  \
	}
#define BPF_JUMP(code, k, jt, jf)                                            \
	{                                                                     \
		(unsigned short)(code), (unsigned char)(jt),                      \
			(unsigned char)(jf), k                                       \
	}
#endif // !BPF_STMT

#endif // __linux__

#endif
