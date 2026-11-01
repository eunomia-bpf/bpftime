#include "rocm_attach_impl.hpp"
#include <cerrno>
#include <spdlog/spdlog.h>

namespace bpftime::attach
{
int rocm_attach_impl::detach_by_id(int)
{
	return -ENOENT;
}

int rocm_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&, const attach_private_data &, int attach_type)
{
	if (attach_type == ATTACH_ROCM_PROBE_AND_RETPROBE)
		SPDLOG_ERROR(
			"ROCm dispatch interception is available through bpftime start --rocm; eBPF payload compilation is not implemented yet");
	else
		SPDLOG_ERROR("Unsupported ROCm attach type: {}", attach_type);
	return -ENOTSUP;
}
} // namespace bpftime::attach
