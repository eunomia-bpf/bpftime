#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include <base_attach_impl.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <fstream>
namespace bpftime
{
namespace attach
{

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;

struct nv_hooker_func_t {
	void *func;
};

struct nv_attach_basic_private_data final : public attach_private_data {
	// function name to probe
	std::string probe_func_name;
	// initialize_from_string
	int initialize_from_string(const std::string_view &sv) override;
};

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl_basic final : public base_attach_impl {
    public:
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	nv_attach_impl_basic(const nv_attach_impl_basic &) = delete;
	nv_attach_impl_basic &operator=(const nv_attach_impl_basic &) = delete;
	nv_attach_impl_basic()
	{
	}
	// Forward declare the nested Impl struct
	struct Impl;

    private:
	// The original syscall function
	nv_hooker_func_t orig_nv = { nullptr };
};

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
