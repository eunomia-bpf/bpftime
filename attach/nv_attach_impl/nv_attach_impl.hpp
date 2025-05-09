#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include "cuda_injector.hpp"
#include "ebpf_inst.h"
#include "nv_attach_utils.hpp"
#include <base_attach_impl.hpp>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <map>
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
#include <variant>
#include <vector>

namespace bpftime
{
namespace attach
{
std::string filter_compiled_ptx_for_ebpf_program(std::string input);

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;

struct nv_hooker_func_t {
	void *func;
};

enum class AttachedToFunction { RegisterFatbin };
struct CUDARuntimeFunctionHookerContext {
	class nv_attach_impl *impl;
	AttachedToFunction to_function;
};

struct nv_attach_cuda_memcapture {};
struct nv_attach_function_probe {
	std::string func;
	bool is_retprobe;
};

using nv_attach_type =
	std::variant<nv_attach_cuda_memcapture, nv_attach_function_probe>;
struct nv_attach_entry {
	std::string probe_ptx;
	nv_attach_type type;
	uintptr_t shared_mem_ptr;
	std::vector<ebpf_inst> instuctions;
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
	std::unique_ptr<CUDAInjector> injector;
	std::vector<std::unique_ptr<__fatBinC_Wrapper_t>> stored_binaries_header;
	std::vector<std::unique_ptr<std::vector<uint8_t>>> stored_binaries_body;
	std::optional<std::vector<uint8_t>>
	hack_fatbin(std::vector<uint8_t> &&);
	std::optional<std::string>
	patch_with_memcapture(std::string, const nv_attach_entry &entry);

    private:
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::set<std::string> to_hook_device_functions;
	std::map<int, nv_attach_entry> hook_entries;
};

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
