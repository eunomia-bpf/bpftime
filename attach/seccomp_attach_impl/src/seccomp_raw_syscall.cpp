/* SPDX-License-Identifier: MIT */
#include "seccomp_raw_syscall.hpp"
#include <cerrno>

#if defined(__linux__) && defined(__x86_64__)

namespace bpftime
{
namespace seccomp
{
int64_t raw_syscall(int64_t nr, int64_t a1, int64_t a2, int64_t a3,
		    int64_t a4, int64_t a5, int64_t a6)
{
	// The seccomp worker is a normal userspace thread, so a direct
	// syscall instruction reaches the kernel without any interposition.
	// Use inline asm rather than the glibc syscall() wrapper so this does
	// not depend on any particular libc exposing that symbol.
	int64_t ret;
	register int64_t r10 __asm__("r10") = a4;
	register int64_t r8 __asm__("r8") = a5;
	register int64_t r9 __asm__("r9") = a6;
	__asm__ volatile("syscall"
			 : "=a"(ret)
			 : "a"(nr), "D"(a1), "S"(a2), "d"(a3),
			   "r"(r10), "r"(r8), "r"(r9)
			 : "rcx", "r11", "memory", "cc");
	return ret;
}
} // namespace seccomp
} // namespace bpftime

#else
namespace bpftime
{
namespace seccomp
{
int64_t raw_syscall(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
		    int64_t)
{
	// Seccomp user-notification support is x86_64-only in this change.
	return -ENOSYS;
}
} // namespace seccomp
} // namespace bpftime
#endif
