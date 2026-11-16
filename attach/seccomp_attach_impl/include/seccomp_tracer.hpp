/* SPDX-License-Identifier: MIT
 *
 * Public entry point for the seccomp user-notification attach mechanism.
 * The transformer/agent calls this to install the tracer; it is a no-op unless
 * BPFTIME_GO_SECCOMP is set, preserving the existing env gate.
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
