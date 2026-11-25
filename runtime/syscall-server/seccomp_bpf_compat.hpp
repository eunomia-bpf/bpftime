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

// Seccomp USER_NOTIF API constants. Older libc kernel-headers hide these from
// <linux/seccomp.h>; the kernel UAPI values are stable. Self-contained so this
// rootless capture needs no header from the former attach/ location.
#include <linux/seccomp.h>
#include <sys/syscall.h>
#ifndef SECCOMP_RET_USER_NOTIF
#define SECCOMP_RET_USER_NOTIF 0x7fc00000U
#endif
#ifndef SECCOMP_FILTER_FLAG_NEW_LISTENER
#define SECCOMP_FILTER_FLAG_NEW_LISTENER (1UL << 3)
#endif
#ifndef SECCOMP_USER_NOTIF_FLAG_CONTINUE
#define SECCOMP_USER_NOTIF_FLAG_CONTINUE (1UL << 0)
#endif
#ifndef SECCOMP_IOCTL_NOTIF_RECV
struct seccomp_notif {
	uint64_t id;
	uint32_t pid;
	uint32_t flags;
	struct seccomp_data data;
};
struct seccomp_notif_resp {
	uint64_t id;
	int64_t val;
	int32_t error;
	uint32_t flags;
};
#define SECCOMP_IOCTL_NOTIF_RECV _IOWR('!', 0, struct seccomp_notif)
#define SECCOMP_IOCTL_NOTIF_SEND _IOWR('!', 1, struct seccomp_notif_resp)
#endif

#endif // __linux__

#endif
