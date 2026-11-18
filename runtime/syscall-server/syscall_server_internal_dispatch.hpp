/* SPDX-License-Identifier: MIT
 *
 * Internal dispatch entry shared inside the bpftime-syscall-server shared
 * object. The seccomp user-notification workers (compiled into this same
 * object after the rootless decoupling) call this directly.
 *
 * ARCHITECTURE INVARIANT — DO NOT BREAK:
 * This header and everything it reaches must stay inside runtime/. The
 * syscall server must never include, link or symbol-reference anything
 * under attach/ (no frida, no capstone, no text-segment transformer, no
 * attach implementation). The only seam toward the outside world is the
 * kernel-ABI-style C function signature below. See runtime/README.md and
 * attach/README.md.
 */
#ifndef BPFTIME_SYSCALL_SERVER_INTERNAL_DISPATCH_HPP
#define BPFTIME_SYSCALL_SERVER_INTERNAL_DISPATCH_HPP

#include <cstdint>

// Emulate one syscall inside the fake kernel. Returns the libc-style result
// (-1 with errno set on failure); callers that answer seccomp notifications
// translate that into a negative errno themselves.
long dispatch_syscall(long sysno, long arg1, long arg2, long arg3, long arg4,
		      long arg5, long arg6) noexcept;

// Run the one-time fake-kernel/SHM initialization before any seccomp filter
// is installed. Calling it lazily from inside a seccomp worker deadlocks: the
// startup path itself issues whitelisted syscalls (close/mmap/...) that the
// not-yet-initialized workers cannot answer. Returns 0 on success. Internal to
// this shared object; not exported.
int bpftime_syscall_server_prewarm() noexcept;
#endif
