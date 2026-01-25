#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include "ebpf_inst.h"
#include "nv_attach_utils.hpp"
#include "ptx_compiler/ptx_compiler.hpp"
#include "ptxpass/core.hpp"
#include <base_attach_impl.hpp>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <nvml.h>
#include <cuda.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include "nv_attach_fatbin_record.hpp"
#include <tuple>
#include <variant>
#include <vector>
#include <atomic>

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

struct nv_attach_hook_state {
	// Active impl instance used by global Frida replace hooks.
	std::atomic<class nv_attach_impl *> active_impl{ nullptr };
	// Original function pointers captured by gum_interceptor_replace.
	std::atomic<void *> orig_cuda_launch_kernel{ nullptr };
	std::atomic<void *> orig_cuda_launch_kernel_ptsz{ nullptr };
	std::atomic<void *> orig_cu_graph_add_kernel_node_v1{ nullptr };
	std::atomic<void *> orig_cu_graph_add_kernel_node_v2{ nullptr };
	std::atomic<void *> orig_cu_graph_exec_kernel_node_set_params_v1{ nullptr };
	std::atomic<void *> orig_cu_graph_exec_kernel_node_set_params_v2{ nullptr };
	std::atomic<void *> orig_cu_graph_kernel_node_set_params_v1{ nullptr };
	std::atomic<void *> orig_cu_graph_kernel_node_set_params_v2{ nullptr };
	std::atomic<void *> orig_cuda_memcpy_from_symbol{ nullptr };
	std::atomic<void *> orig_cuda_memcpy_from_symbol_async{ nullptr };

	// Whether Frida replace hooks have been installed in this process.
	std::atomic<bool> replacements_installed{ false };
};

nv_attach_hook_state &nv_attach_get_hook_state();
void nv_attach_set_active_impl(class nv_attach_impl *impl);
class nv_attach_impl *nv_attach_get_active_impl();

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
	std::optional<std::map<std::string, std::tuple<std::string, bool>>>
		hack_fatbin(std::map<std::string, std::string>);
	std::map<std::string, std::string>
	extract_ptxs(std::vector<uint8_t> &&);
	void mirror_cuda_memcpy_to_symbol(const void *symbol, const void *src,
					  size_t count, size_t offset,
					  cudaMemcpyKind kind,
					  cudaStream_t stream, bool async);
	void mirror_cuda_memcpy_from_symbol(void *dst, const void *symbol,
					    size_t count, size_t offset,
					    cudaMemcpyKind kind,
					    cudaStream_t stream, bool async);

	int find_attach_entry_by_program_name(const char *name) const;
	int run_attach_entry_on_gpu(int attach_id, int run_count = 1,
				    int grid_dim_x = 1, int grid_dim_y = 1,
				    int grid_dim_z = 1, int block_dim_x = 1,
				    int block_dim_y = 1, int block_dim_z = 1);
	void record_patched_kernel_function(const std::string &kernel_name,
					    CUfunction function);
	std::optional<CUfunction>
	find_patched_kernel_function(const std::string &kernel_name) const;
	// Notify nv_attach_impl that a patched kernel launch was enqueued on a
	// stream. Used to coordinate detach with in-flight patched kernels so we
	// don't tear down loader-owned CUDA IPC buffers prematurely.
	void record_patched_launch(cudaStream_t stream);
	void record_original_cufunction_name(CUfunction function,
					     const std::string &kernel_name);
	std::optional<std::string>
	find_original_kernel_name(CUfunction function) const;
			// Late attach support: attempt to discover already-loaded CUDA fatbins and
		// prefill patched kernel mappings.
		void bootstrap_existing_fatbins_once();
		void start_late_bootstrap_async();
		bool is_late_bootstrap_done() const noexcept
		{
			return late_bootstrap_done.load(std::memory_order_acquire);
		}
		// Resolve host-side kernel stub to symbol name. Uses dladdr first, then a
		// cached ELF symbol table fallback.
		std::optional<std::string> resolve_host_function_symbol(void *addr);
	// Whether nv_attach is currently enabled (can be disabled by detach).
	bool is_enabled() const noexcept;
	std::vector<std::unique_ptr<fatbin_record>> fatbin_records;
	fatbin_record *current_fatbin = nullptr;
	std::map<void *, fatbin_record *> symbol_address_to_fatbin;
	uintptr_t shared_mem_ptr;
	std::optional<std::vector<MapBasicInfo>> map_basic_info;
	void *ptx_compiler_dl_handle = nullptr;
	nv_attach_impl_ptx_compiler_handler ptx_compiler;
	/// SHA256 of ELF -> PTX module
	std::shared_ptr<std::map<std::string, std::shared_ptr<ptx_in_module>>>
		module_pool;
	/// SHA256 of PTX -> ELF
	std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> ptx_pool;

	// Original function pointers for Frida replace hooks (trampolines)
	// They are set by gum_interceptor_replace(...) and must be used to call
	// the original implementation (calling the symbol directly will
	// recurse). Which is used for cudagraph hook.
	void *original_cuda_launch_kernel = nullptr;
	void *original_cuda_launch_kernel_ptsz = nullptr;
	void *original_cu_graph_add_kernel_node_v1 = nullptr;
	void *original_cu_graph_add_kernel_node_v2 = nullptr;
	void *original_cu_graph_exec_kernel_node_set_params_v1 = nullptr;
	void *original_cu_graph_exec_kernel_node_set_params_v2 = nullptr;
	void *original_cu_graph_kernel_node_set_params_v1 = nullptr;
	void *original_cu_graph_kernel_node_set_params_v2 = nullptr;
	void *original_cuda_memcpy_from_symbol = nullptr;
	void *original_cuda_memcpy_from_symbol_async = nullptr;

	    private:
		void record_patched_launch_event(CUstream stream);
		void wait_for_patched_launch_events(std::chrono::milliseconds timeout);
		void clear_patched_state_for_next_session();

		void bootstrap_existing_fatbins();
		void reset_late_bootstrap_state_for_next_attach();
		void build_host_symbol_cache_once();
		void prefill_patched_kernel_functions_from_loaded_fatbins();
		std::vector<std::string> collect_all_kernels_to_patch() const;

	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::map<int, nv_attach_entry> hook_entries;
	// discovered pass definitions
	std::vector<std::unique_ptr<pass_cfg_with_exec_path>>
		pass_configurations;
	std::map<std::string, ptxpass::runtime_response::RuntimeResponse>
		patch_cache;
	mutable std::mutex cuda_symbol_map_mutex;
	std::unordered_map<std::string, CUfunction> patched_kernel_by_name;
	std::unordered_map<CUfunction, std::string> kernel_name_by_cufunction;

			std::atomic<bool> enabled{ true };
			// Late bootstrap needs to be repeatable across trace sessions.
			// Using a heap-allocated once_flag allows resetting it after detach.
			std::unique_ptr<std::once_flag> late_bootstrap_once =
				std::make_unique<std::once_flag>();
			std::atomic<bool> late_bootstrap_started{ false };
			std::atomic<bool> late_bootstrap_done{ false };
			std::mutex late_bootstrap_mutex;
			std::once_flag host_symbol_cache_once;
			mutable std::mutex host_symbol_cache_mutex;
	// Absolute address sorted list of host function symbols across loaded
	// modules (best-effort).
	struct host_symbol_range {
		std::uintptr_t start = 0;
		std::uintptr_t end = 0;
		std::string name;
	};
	std::vector<host_symbol_range> host_symbol_ranges;

		mutable std::mutex patched_global_cache_mutex;
		std::unordered_map<std::string, std::pair<CUdeviceptr, size_t>>
			patched_global_by_name;

		mutable std::mutex launch_event_mutex;
		std::unordered_map<CUstream, CUevent> pending_launch_events_by_stream;
	};

std::string add_semicolon_for_variable_lines(std::string input);
} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
