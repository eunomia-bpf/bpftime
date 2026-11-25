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

#elif defined(__linux__) && defined(__aarch64__)

namespace bpftime
{
namespace seccomp
{
int64_t raw_syscall(int64_t nr, int64_t a1, int64_t a2, int64_t a3,
		    int64_t a4, int64_t a5, int64_t a6)
{
	// aarch64 Linux syscall ABI: the number goes in x8, the six arguments
	// in x0..x5 and the result comes back in x0. Inline asm rather than the
	// glibc syscall() wrapper keeps the worker independent of libc and
	// avoids re-entering any interposed symbol.
	register int64_t x8 __asm__("x8") = nr;
	register int64_t x0 __asm__("x0") = a1;
	register int64_t x1 __asm__("x1") = a2;
	register int64_t x2 __asm__("x2") = a3;
	register int64_t x3 __asm__("x3") = a4;
	register int64_t x4 __asm__("x4") = a5;
	register int64_t x5 __asm__("x5") = a6;
	__asm__ volatile("svc #0"
			 : "=r"(x0)
			 : "r"(x8), "0"(x0), "r"(x1), "r"(x2), "r"(x3),
			   "r"(x4), "r"(x5)
			 : "memory", "cc");
	return x0;
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
	// Seccomp user-notification support exists only on x86_64/aarch64.
	return -ENOSYS;
}
} // namespace seccomp
} // namespace bpftime
#endif
