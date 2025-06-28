#ifndef _BPFTIME_ROCM_ATTACH_IMPL_HPP
#define _BPFTIME_ROCM_ATTACH_IMPL_HPP

#include "base_attach_impl.hpp"
namespace bpftime
{
namespace attach
{

constexpr int ATTACH_ROCM_PROBE_AND_RETPROBE = 1018;

enum class RocmAttachedToFunction {
	RegisterFatbin,
	RegisterFunction,
	RegisterFatbinEnd,
	HipLaunchKernel
};
struct ROCMRuntimeFunctionHookerContext {
	class rocm_attach_impl *impl;
	RocmAttachedToFunction to_function;
};
class rocm_attach_impl final : public base_attach_impl {
    public:
	rocm_attach_impl();
	virtual ~rocm_attach_impl();
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	rocm_attach_impl(const rocm_attach_impl &) = delete;
	rocm_attach_impl &operator=(const rocm_attach_impl &) = delete;

    private:
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<ROCMRuntimeFunctionHookerContext>>
		hooker_contexts;
};
} // namespace attach
} // namespace bpftime

#endif
