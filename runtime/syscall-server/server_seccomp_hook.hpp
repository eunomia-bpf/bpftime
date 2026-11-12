/* SPDX-License-Identifier: MIT
 *
 * Self-contained seccomp user-notification hook for the syscall server.
 *
 * The server's LD_PRELOAD libc overrides cannot intercept a Go runtime that
 * issues raw syscall instructions. This hook installs a kernel seccomp
 * user-notification filter inside the server process; notified syscalls are
 * emulated by the server's own dispatcher. It deliberately includes no
 * attach/ header: the server must remain buildable without the agent.
 */
#ifndef BPFTIME_SERVER_SECCOMP_HOOK_HPP
#define BPFTIME_SECCOMP_HOOK_HPP

namespace bpftime
{
namespace server
{
// Install the seccomp user-notification hook. No-op unless
// BPFTIME_SERVER_SECCOMP is set. Returns false on install failure.
bool install_server_seccomp_hook();
} // namespace server
} // namespace bpftime

#endif
