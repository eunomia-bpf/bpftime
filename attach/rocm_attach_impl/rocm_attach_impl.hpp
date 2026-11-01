#ifndef _BPFTIME_ROCM_ATTACH_IMPL_HPP
#define _BPFTIME_ROCM_ATTACH_IMPL_HPP

#include "base_attach_impl.hpp"

namespace bpftime::attach
{
constexpr int ATTACH_ROCM_PROBE_AND_RETPROBE = 1018;

class rocm_attach_impl final : public base_attach_impl {
    public:
	int detach_by_id(int id) override;
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type) override;
};
} // namespace bpftime::attach
#endif
