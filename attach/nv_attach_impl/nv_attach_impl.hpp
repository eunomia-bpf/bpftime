#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include "ebpf_inst.h"
#include "nv_attach_utils.hpp"
#include "ptxpass/core.hpp"
#include <base_attach_impl.hpp>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
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
#include "nv_attach_fatbin_record.hpp"
#include <tuple>
#include <variant>
#include <vector>

namespace bpftime
{
namespace attach
{

using print_config_fn = void (*)(int length, char *out);
using process_input_fn = int (*)(const char *input, int length, char *output);

std::string filter_compiled_ptx_for_ebpf_program(std::string input,
						 std::string);

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
	RegisterVariable,
	RegisterFatbinEnd,
	CudaMalloc,
	CudaMallocManaged,
	CudaMemcpyToSymbol,
	CudaMemcpyToSymbolAsync
};
struct CUDARuntimeFunctionHookerContext {
	class nv_attach_impl *impl;
	AttachedToFunction to_function;
};

struct nv_attach_entry {
	std::vector<ebpf_inst> instuctions;
	// Kernels to be patched for this attach entry
	std::vector<std::string> kernels;
	// program name for this attach entry
	std::string program_name;
	// pass-based execution fields
	std::map<std::string, std::string> parameters; // arbitrary parameters
						       // for pass
	// Extra serialized parameters (JSON string) reserved for future use
	std::optional<std::string> extras;
	struct pass_cfg_with_exec_path *config;
};

struct pass_cfg_with_exec_path {
	std::filesystem::path executable_path;
	ptxpass::pass_config::PassConfig pass_config;
	print_config_fn print_config;
	process_input_fn process_input;

	void *handle;

	pass_cfg_with_exec_path(std::filesystem::path path,
				ptxpass::pass_config::PassConfig config,
				print_config_fn print_config,
				process_input_fn process_input, void *handle)
		: executable_path(path), pass_config(config),
		  print_config(print_config), process_input(process_input),
		  handle(handle)
	{
	}

	pass_cfg_with_exec_path(const pass_cfg_with_exec_path &) = delete;
	pass_cfg_with_exec_path &
	operator=(const pass_cfg_with_exec_path &) = delete;
	pass_cfg_with_exec_path(pass_cfg_with_exec_path &&) = default;
	pass_cfg_with_exec_path &
	operator=(pass_cfg_with_exec_path &&) = default;

	~pass_cfg_with_exec_path()
	{
		dlclose(handle);
	}
};

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	int detach_by_id(int id) override;
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type) override;
	// Register CUDA-specific ext helpers required by LLVM-JIT to resolve
	// symbols like _bpf_helper_ext_0502/_0503 when compiling programs
	void register_custom_helpers(
		ebpf_helper_register_callback register_callback) override;
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl();
	virtual ~nv_attach_impl();
	std::optional<std::map<std::string, std::string>>
		hack_fatbin(std::map<std::string, std::string>);
	std::map<std::string, std::string>
	extract_ptxs(std::vector<uint8_t> &&);
	void mirror_cuda_memcpy_to_symbol(const void *symbol, const void *src,
					  size_t count, size_t offset,
					  cudaMemcpyKind kind,
					  cudaStream_t stream, bool async);

	int find_attach_entry_by_program_name(const char *name) const;
	int run_attach_entry_on_gpu(int attach_id, int run_count = 1,
				    int grid_dim_x = 1, int grid_dim_y = 1,
				    int grid_dim_z = 1, int block_dim_x = 1,
				    int block_dim_y = 1, int block_dim_z = 1);
	std::vector<std::unique_ptr<fatbin_record>> fatbin_records;
	fatbin_record *current_fatbin = nullptr;
	std::map<void *, fatbin_record *> symbol_address_to_fatbin;
	uintptr_t shared_mem_ptr;
	std::optional<std::vector<MapBasicInfo>> map_basic_info;

    private:
	void rebase_gpu_ringbuf_map_buffers();
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::map<int, nv_attach_entry> hook_entries;
	// discovered pass definitions
	std::vector<std::unique_ptr<pass_cfg_with_exec_path>>
		pass_configurations;
};

std::string add_semicolon_for_variable_lines(std::string input);
} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
