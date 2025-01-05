#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include <base_attach_impl.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
namespace bpftime
{
namespace attach
{
extern std::optional<class nv_attach_impl *> global_nv_attach_impl;
struct nv_hooker_func_t {
	void *func;
};

struct nv_attach_private_data final : public attach_private_data {
	// The address to hook
	uint64_t addr;
	// Saved module name
	pid_t pid;
};
// Used by text segment transformer to setup syscall callback
// Text segment transformer should provide a pointer to its syscall executor
// function. syscall trace attach impl will save the original syscall function,
// and replace it with one that was handled by syscall trace attach impl
extern "C" void _bpftime__setup_nv_hooker_callback(nv_hooker_func_t *hooker);

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	// Dispatch a syscall from text transformer
	int64_t dispatch_nv(int64_t arg1, int64_t arg2, int64_t arg3,
			    int64_t arg4, int64_t arg5, int64_t arg6);
	// Set the function of calling original nv
	void set_original_nv_function(nv_hooker_func_t func)
	{
		orig_nv = func;
	}
	// Set this nv trace attach impl instance to the global ones, which
	// could be accessed by text segment transformer
	void set_to_global()
	{
		global_nv_attach_impl = this;
	}
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl()
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
