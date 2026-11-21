/* SPDX-License-Identifier: MIT
 *
 * Seccomp attach implementation: issue the real syscall without depending
 * on the Frida text-segment trampoline. The seccomp worker runs on its own
 * thread after the kernel has already returned control to userspace, so a
 * plain syscall instruction is sufficient; no binary-rewrite trampoline is
 * involved.
 */
#ifndef BPFTIME_SECCOMP_RAW_SYSCALL_HPP
#define BPFTIME_SECCOMP_RAW_SYSCALL_HPP

#include <cstdint>

namespace bpftime
{
namespace seccomp
{
// Invoke a raw syscall with the native Linux syscall ABI (x86_64 `syscall`
// or aarch64 `svc #0`). Declared for all builds; defined on Linux x86_64 and
// aarch64, matching the seccomp-user-notif gate.
int64_t raw_syscall(int64_t nr, int64_t a1, int64_t a2, int64_t a3,
		    int64_t a4, int64_t a5, int64_t a6);
} // namespace seccomp
} // namespace bpftime

#endif
