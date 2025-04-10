#include "nv_attach_impl_basic.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <optional>
#include <stdexcept>

#ifdef __linux__
#include <asm/unistd.h> // For architecture-specific syscall numbers
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <cstdio>
#include <cstring>

// ------------------------------------------------------------------
// Implementation of the nv_attach_impl interface
// ------------------------------------------------------------------
namespace bpftime
{
namespace attach
{

int nv_attach_impl_basic::detach_by_id(int id)
{
	return 0;
}

int nv_attach_impl_basic::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	return 0;
}

int nv_attach_basic_private_data::initialize_from_string(const std::string_view &sv)
{
	SPDLOG_INFO("nv_attach_private_data::initialize_from_string({})", sv);
	if (sv != "vprintf") {
		throw std::runtime_error(
			"For demo purpose, nv_attach_impl only supports probing vprintf");
	}
	this->probe_func_name = sv;
	return 0;
}

} // namespace attach
} // namespace bpftime
