#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
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

// #include <pos/include/oob/ckpt_dump.h>
#include <variant>
#include <vector>

namespace bpftime
{
namespace attach
{
std::string filter_compiled_ptx_for_ebpf_program(std::string input,
						 std::string);
std::string add_register_guard_for_ebpf_ptx_func(const std::string &ptxCode);

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;
struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
	int map_type;
	void *extra_buffer;
	uint64_t max_thread_count;
};
struct nv_hooker_func_t {
	void *func;
};

enum class AttachedToFunction {
	RegisterFatbin,
	RegisterFunction,
	RegisterFatbinEnd,
	CudaLaunchKernel
};
enum class TrampolineMemorySetupStage { NotSet, Registered, Copied };
struct CUDARuntimeFunctionHookerContext {
	class nv_attach_impl *impl;
	AttachedToFunction to_function;
};

struct nv_attach_cuda_memcapture {};
struct nv_attach_function_probe {
	std::string func;
	bool is_retprobe;
};
struct nv_attach_directly_run_on_gpu {};
using nv_attach_type =
	std::variant<nv_attach_cuda_memcapture, nv_attach_function_probe,
		     nv_attach_directly_run_on_gpu>;
struct nv_attach_entry {
	nv_attach_type type;
	std::vector<ebpf_inst> instuctions;
	// Kernels to be patched for this attach entry
	std::vector<std::string> kernels;
	// program name for this attach entry
	std::string program_name;
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
	std::vector<std::unique_ptr<__fatBinC_Wrapper_t>> stored_binaries_header;
	std::vector<std::unique_ptr<std::vector<uint8_t>>> stored_binaries_body;
	std::optional<std::vector<uint8_t>>
	hack_fatbin(std::vector<uint8_t> &&);
	std::optional<std::string>
	patch_with_memcapture(std::string, const nv_attach_entry &entry,
			      bool should_set_trampoline);
	std::optional<std::string>
	patch_with_probe_and_retprobe(std::string, const nv_attach_entry &,
				      bool should_set_trampoline);
	int register_trampoline_memory(void **);
	int copy_data_to_trampoline_memory();
	TrampolineMemorySetupStage trampoline_memory_state =
		TrampolineMemorySetupStage::NotSet;
	int find_attach_entry_by_program_name(const char *name) const;
	int run_attach_entry_on_gpu(int attach_id, int run_count = 1,
				    int grid_dim_x = 1, int grid_dim_y = 1,
				    int grid_dim_z = 1, int block_dim_x = 1,
				    int block_dim_y = 1, int block_dim_z = 1);

    private:
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::set<std::string> to_hook_device_functions;
	std::map<int, nv_attach_entry> hook_entries;
	uintptr_t shared_mem_ptr;
	std::optional<std::vector<MapBasicInfo>> map_basic_info;
};
std::string filter_unprintable_chars(std::string input);
std::string filter_out_version_headers(const std::string &input);
std::string
generate_ptx_for_ebpf(const std::vector<ebpf_inst> &inst,
		      const std::string &func_name, bool with_arguments,
		      bool add_register_guard_and_filter_version_headers);

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
