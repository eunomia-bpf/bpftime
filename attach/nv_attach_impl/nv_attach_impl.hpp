#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include "cuda_injector.hpp"
#include <base_attach_impl.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <memory>
#include <nvml.h>
#include <cuda.h>
#include <optional>
#include <set>
#include <string>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>

#include <pos/include/oob/ckpt_dump.h>
#include <vector>

namespace bpftime
{
namespace attach
{

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;
template <class T>
static inline T try_get_original_func(const char *name, T &store)
{
	if (store == nullptr) {
		store = (T)dlsym(RTLD_NEXT, name);
	}
	return store;
}

struct nv_hooker_func_t {
	void *func;
};

enum class AttachedToFunction { RegisterFatbin };
struct CUDARuntimeFunctionHookerContext {
	class nv_attach_impl *impl;
	AttachedToFunction to_function;
};

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl();
	virtual ~nv_attach_impl();

    private:
	std::optional<CUDAInjector> injector;
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::set<std::string> to_hook_device_functions;
	std::vector<std::unique_ptr<std::vector<char>>> stored_binaries;
};

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
