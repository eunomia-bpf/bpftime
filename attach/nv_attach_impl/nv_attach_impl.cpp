#include "nv_attach_impl.hpp"
#include "cuda_injector.hpp"
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

int nv_attach_impl::detach_by_id(int id)
{
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	return 0;
}

} // namespace attach
} // namespace bpftime
