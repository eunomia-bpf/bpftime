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
// Invoke a raw syscall with the classic x86_64 syscall ABI. Declared for all
// builds; defined only on Linux x86_64, matching the seccomp-user-notif gate.
int64_t raw_syscall(int64_t nr, int64_t a1, int64_t a2, int64_t a3,
		    int64_t a4, int64_t a5, int64_t a6);
} // namespace seccomp
} // namespace bpftime

#endif
