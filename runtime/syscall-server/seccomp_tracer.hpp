/* SPDX-License-Identifier: MIT
 *
 * Entry point for the seccomp USER_NOTIF tracer installed by the rootless
 * syscall-server (loader constructor and __libc_start_main interposer); it is
 * a no-op unless BPFTIME_GO_SECCOMP is set, preserving the existing env gate.
 */
#ifndef BPFTIME_SECCOMP_TRACER_HPP
#define BPFTIME_SECCOMP_TRACER_HPP

namespace bpftime
{
namespace seccomp
{
// Install the seccomp user-notification tracer. Returns false if the env
// marker is absent or the filter/listener could not be installed.
bool install_seccomp_tracer();
} // namespace seccomp
} // namespace bpftime

#endif
