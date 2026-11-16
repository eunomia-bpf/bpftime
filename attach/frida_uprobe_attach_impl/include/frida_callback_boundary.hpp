#ifndef _BPFTIME_FRIDA_CALLBACK_BOUNDARY_HPP
#define _BPFTIME_FRIDA_CALLBACK_BOUNDARY_HPP

#include "spdlog/spdlog.h"
#include <exception>
#include <utility>

namespace bpftime
{
namespace attach
{
// Single exception boundary for every entry point invoked from frida-gum C
// frames (the replacement function and the GumInvocationListener callbacks).
//
// A C++ exception that escapes one of those C frames cannot be unwound and
// would call std::terminate, killing the host process. This helper is
// therefore noexcept: it runs the supplied callback and, on any exception,
// logs the failure and invokes the fail-safe fallback instead of letting the
// exception cross the C boundary. The fallback must itself be non-throwing.
//
// This mirrors the syscall-server interposer boundary
// (handle_exceptions in runtime/syscall-server/syscall_server_main.cpp) and
// is the frida-side realization of the L1 injection-boundary contract.
template <typename Fn, typename Fallback>
auto frida_c_boundary(Fn &&fn, Fallback &&fallback) noexcept
	-> decltype(fallback())
{
	try {
		return fn();
	} catch (const std::exception &e) {
		SPDLOG_ERROR(
			"bpftime frida callback threw an exception, running the fail-safe fallback: {}",
			e.what());
		return fallback();
	} catch (...) {
		SPDLOG_ERROR(
			"bpftime frida callback threw an unknown exception, running the fail-safe fallback");
		return fallback();
	}
}

} // namespace attach
} // namespace bpftime

#endif // _BPFTIME_FRIDA_CALLBACK_BOUNDARY_HPP
