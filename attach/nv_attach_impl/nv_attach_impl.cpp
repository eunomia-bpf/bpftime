#include "nv_attach_impl.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "ebpf_inst.h"
#include "frida-gum.h"

#include "nvPTXCompiler.h"
#include "nv_attach_private_data.hpp"
#include "nv_attach_utils.hpp"
#include "ptx_compiler/ptx_compiler.hpp"
#include "nv_elf_introspect.hpp"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers
#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/process.hpp>
#include <boost/process/detail/child_decl.hpp>
#include <boost/process/env.hpp>
#include <boost/process/io.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/start_dir.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/ptrace.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#if __linux__
#include <spawn.h>
#endif
#include <link.h>
#include <tuple>
#include <unistd.h>
#include <variant>
#include <vector>
#include <atomic>
#include <optional>
#include <thread>
#include "ptxpass/core.hpp"
#include "ptx_pass_config.h"

#if __linux__
extern "C" char **environ;
#endif

using namespace bpftime;
using namespace attach;

namespace bpftime::attach
{
static nv_attach_hook_state g_nv_attach_hook_state{};

nv_attach_hook_state &nv_attach_get_hook_state()
{
	return g_nv_attach_hook_state;
}

void nv_attach_set_active_impl(nv_attach_impl *impl)
{
	g_nv_attach_hook_state.active_impl.store(impl,
						 std::memory_order_release);
}

nv_attach_impl *nv_attach_get_active_impl()
{
	return g_nv_attach_hook_state.active_impl.load(
		std::memory_order_acquire);
}
} // namespace bpftime::attach

static std::vector<std::filesystem::path> split_by_colon(const std::string &str)
{
	std::vector<std::filesystem::path> result;

	char *buffer = new char[str.length() + 1];
	strcpy(buffer, str.c_str());

	char *token = strtok(buffer, ":");
	while (token != nullptr) {
		result.push_back(token);
		token = strtok(nullptr, ":");
	}

	delete[] buffer;
	return result;
}
#define CUDA_DRIVER_CHECK_NO_EXCEPTION(expr, message)                          \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
		}                                                              \
	} while (false)

extern GType cuda_runtime_function_hooker_get_type();

int nv_attach_impl::detach_by_id(int id)
{
	auto itr = hook_entries.find(id);
	if (itr == hook_entries.end()) {
		SPDLOG_WARN("nv_attach_impl: detach unknown id {}", id);
		return -ENOENT;
	}
	hook_entries.erase(itr);
	if (hook_entries.empty()) {
		SPDLOG_INFO(
			"nv_attach_impl: last entry detached; disabling CUDA launch replacement");
		enabled.store(false, std::memory_order_release);
		wait_for_patched_launch_events(std::chrono::seconds(2));
		clear_patched_state_for_next_session();
		reset_late_bootstrap_state_for_next_attach();
	} else {
		SPDLOG_INFO("nv_attach_impl: detached id {}", id);
	}
	return 0;
}

void nv_attach_impl::register_custom_helpers(
	ebpf_helper_register_callback register_callback)
{
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	auto data = dynamic_cast<const nv_attach_private_data &>(private_data);

	if (!std::holds_alternative<std::string>(data.code_addr_or_func_name)) {
		SPDLOG_ERROR(
			"code_addr_or_func_name does not hold a string value");
		return -1;
	}
	const auto &func_name =
		std::get<std::string>(data.code_addr_or_func_name);
	std::string attach_point_name;
	if (attach_type == ATTACH_CUDA_PROBE) {
		attach_point_name = "kprobe/" + func_name;
	} else if (attach_type == ATTACH_CUDA_RETPROBE) {
		attach_point_name = "kretprobe/" + func_name;
	} else {
		attach_point_name = func_name;
	}
	struct pass_cfg_with_exec_path *matched = nullptr;
	for (const auto &pd : this->pass_configurations) {
		if (pd->pass_config.attach_type != attach_type)
			continue;
		ptxpass::AttachPointMatcher matcher(
			pd->pass_config.attach_points);

		if (matcher.matches(attach_point_name)) {
			matched = pd.get();
			break;
		}
	}
	if (matched) {
		enabled.store(true, std::memory_order_release);
		auto id = this->allocate_id();
		nv_attach_entry entry;
		entry.instuctions = data.instructions;
		entry.kernels = data.func_names;
		entry.program_name = data.program_name;
		entry.config = matched;

		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		if (data.comm_shared_mem == 0) {
			SPDLOG_ERROR(
				"comm_shared_mem is null when creating CUDA attach for {}",
				func_name);
			return -1;
		}
		if (this->shared_mem_ptr != data.comm_shared_mem) {
			this->shared_mem_ptr = data.comm_shared_mem;
			SPDLOG_INFO("Cached shared_mem_ptr at {:x}",
				    (uintptr_t)this->shared_mem_ptr);
		}
		SPDLOG_INFO("Recorded pass {} for func {}",
			    matched->executable_path.c_str(), func_name);
		start_late_bootstrap_async();
		return id;
	}
	SPDLOG_WARN(
		"No pass definition matched for function {}, attach_type {}. Skipping.",
		attach_point_name, attach_type);
	return -1;
}

extern "C" {
cudaError_t cuda_runtime_function__cudaLaunchKernel(const void *func,
						    dim3 gridDim, dim3 blockDim,
						    void **args,
						    size_t sharedMem,
						    cudaStream_t stream);
cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz(
	const void *func, dim3 gridDim, dim3 blockDim, void **args,
	size_t sharedMem, cudaStream_t stream);
CUresult cuda_driver_function__cuGraphAddKernelNode_v1(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphAddKernelNode_v2(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v1(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v2(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
cudaError_t cuda_runtime_function__cudaMemcpyFromSymbol(
	void *dst, const void *symbol, size_t count, size_t offset = 0,
	cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
cudaError_t cuda_runtime_function__cudaMemcpyFromSymbolAsync(
	void *dst, const void *symbol, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream = 0);
}

nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	{
		void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
		if (handle == nullptr) {
			SPDLOG_DEBUG("dlopen(libcuda.so.1) failed: {}",
				     dlerror());
		}
	}
	this->module_pool = std::make_shared<
		std::map<std::string, std::shared_ptr<ptx_in_module>>>();
	this->ptx_pool =
		std::make_shared<std::map<std::string, std::vector<uint8_t>>>();

	this->shared_mem_ptr = 0;
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	if (interceptor == nullptr) {
		SPDLOG_ERROR("Failed to obtain Frida interceptor");
		throw std::runtime_error(
			"Failed to initialize Frida interceptor");
	}
	auto listener =
		g_object_new(cuda_runtime_function_hooker_get_type(), nullptr);
	if (listener == nullptr) {
		SPDLOG_ERROR("Failed to create Frida listener");
		throw std::runtime_error("Failed to initialize Frida listener");
	}
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;

	auto &hook_state = nv_attach_get_hook_state();
	hook_state.replacements_installed.store(true,
						std::memory_order_release);

	gum_interceptor_begin_transaction(interceptor);

	auto register_hook = [&](AttachedToFunction func, void *addr) {
		if (addr == nullptr) {
			SPDLOG_WARN(
				"Skipping hook registration for function {} - symbol not found",
				(int)func);
			return;
		}
		auto ctx = std::make_unique<CUDARuntimeFunctionHookerContext>();
		ctx->to_function = func;
		ctx->impl = this;
		auto ctx_ptr = ctx.get();
		this->hooker_contexts.push_back(std::move(ctx));
		if (auto result = gum_interceptor_attach(
			    interceptor, (gpointer)addr,
			    (GumInvocationListener *)listener, ctx_ptr);
		    result != GUM_ATTACH_OK) {
			SPDLOG_ERROR(
				"Unable to attach to CUDA functions: func={}, err={}",
				(int)func, (int)result);
			throw std::runtime_error(
				"Failed to attach to CUDA function");
		}
	};
	auto replace_hook_once = [&](const char *symbol_name,
				     gpointer replacement,
				     std::atomic<void *> &original_slot) {
		void *addr = GSIZE_TO_POINTER(
			gum_module_find_export_by_name(nullptr, symbol_name));
		if (addr == nullptr) {
			SPDLOG_DEBUG(
				"Skipping replace hook for {} - symbol not found",
				symbol_name);
			return;
		}
		if (original_slot.load(std::memory_order_acquire) != nullptr) {
			return;
		}
		gpointer original_tmp = nullptr;
		auto err = gum_interceptor_replace(interceptor, addr,
						   replacement, &hook_state,
						   (gpointer *)&original_tmp);
		if (err == GUM_REPLACE_OK) {
			original_slot.store(original_tmp,
					    std::memory_order_release);
			return;
		}
		SPDLOG_ERROR("Unable to replace {}: {}", symbol_name, (int)err);
	};

	{
		void *register_fatbin_addr =
			dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
		register_hook(AttachedToFunction::RegisterFatbin,
			      register_fatbin_addr);
	}
	{
		void *register_function_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterFunction"));
		register_hook(AttachedToFunction::RegisterFunction,
			      register_function_addr);
	}
	{
		void *register_variable_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterVar"));
		register_hook(AttachedToFunction::RegisterVariable,
			      register_variable_addr);
	}
	{
		void *register_fatbin_end_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterFatBinaryEnd"));
		register_hook(AttachedToFunction::RegisterFatbinEnd,
			      register_fatbin_end_addr);
	}

	{
		void *cuda_malloc_addr = GSIZE_TO_POINTER(
			gum_module_find_export_by_name(nullptr, "cudaMalloc"));
		register_hook(AttachedToFunction::CudaMalloc, cuda_malloc_addr);
	}
	{
		void *cuda_malloc_managed_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMallocManaged"));
		register_hook(AttachedToFunction::CudaMallocManaged,
			      cuda_malloc_managed_addr);
	}
	{
		void *cuda_memcpy_to_symbol_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMemcpyToSymbol"));
		register_hook(AttachedToFunction::CudaMemcpyToSymbol,
			      cuda_memcpy_to_symbol_addr);
	}
	{
		void *cuda_memcpy_to_symbol_async_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMemcpyToSymbolAsync"));
		register_hook(AttachedToFunction::CudaMemcpyToSymbolAsync,
			      cuda_memcpy_to_symbol_async_addr);
	}
	replace_hook_once("cudaLaunchKernel",
			  (gpointer)&cuda_runtime_function__cudaLaunchKernel,
			  hook_state.orig_cuda_launch_kernel);
	replace_hook_once(
		"cudaLaunchKernel_ptsz",
		(gpointer)&cuda_runtime_function__cudaLaunchKernel_ptsz,
		hook_state.orig_cuda_launch_kernel_ptsz);
	replace_hook_once(
		"cuGraphAddKernelNode",
		(gpointer)&cuda_driver_function__cuGraphAddKernelNode_v1,
		hook_state.orig_cu_graph_add_kernel_node_v1);
	replace_hook_once(
		"cuGraphAddKernelNode_v2",
		(gpointer)&cuda_driver_function__cuGraphAddKernelNode_v2,
		hook_state.orig_cu_graph_add_kernel_node_v2);
	replace_hook_once(
		"cuGraphExecKernelNodeSetParams",
		(gpointer)&cuda_driver_function__cuGraphExecKernelNodeSetParams_v1,
		hook_state.orig_cu_graph_exec_kernel_node_set_params_v1);
	replace_hook_once(
		"cuGraphExecKernelNodeSetParams_v2",
		(gpointer)&cuda_driver_function__cuGraphExecKernelNodeSetParams_v2,
		hook_state.orig_cu_graph_exec_kernel_node_set_params_v2);
	replace_hook_once(
		"cuGraphKernelNodeSetParams",
		(gpointer)&cuda_driver_function__cuGraphKernelNodeSetParams_v1,
		hook_state.orig_cu_graph_kernel_node_set_params_v1);
	replace_hook_once(
		"cuGraphKernelNodeSetParams_v2",
		(gpointer)&cuda_driver_function__cuGraphKernelNodeSetParams_v2,
		hook_state.orig_cu_graph_kernel_node_set_params_v2);
	replace_hook_once(
		"cudaMemcpyFromSymbol",
		(gpointer)&cuda_runtime_function__cudaMemcpyFromSymbol,
		hook_state.orig_cuda_memcpy_from_symbol);
	replace_hook_once(
		"cudaMemcpyFromSymbolAsync",
		(gpointer)&cuda_runtime_function__cudaMemcpyFromSymbolAsync,
		hook_state.orig_cuda_memcpy_from_symbol_async);
	gum_interceptor_end_transaction(interceptor);

	nv_attach_set_active_impl(this);
	this->original_cuda_launch_kernel =
		hook_state.orig_cuda_launch_kernel.load(
			std::memory_order_acquire);
	this->original_cuda_launch_kernel_ptsz =
		hook_state.orig_cuda_launch_kernel_ptsz.load(
			std::memory_order_acquire);
	this->original_cu_graph_add_kernel_node_v1 =
		hook_state.orig_cu_graph_add_kernel_node_v1.load(
			std::memory_order_acquire);
	this->original_cu_graph_add_kernel_node_v2 =
		hook_state.orig_cu_graph_add_kernel_node_v2.load(
			std::memory_order_acquire);
	this->original_cu_graph_exec_kernel_node_set_params_v1 =
		hook_state.orig_cu_graph_exec_kernel_node_set_params_v1.load(
			std::memory_order_acquire);
	this->original_cu_graph_exec_kernel_node_set_params_v2 =
		hook_state.orig_cu_graph_exec_kernel_node_set_params_v2.load(
			std::memory_order_acquire);
	this->original_cu_graph_kernel_node_set_params_v1 =
		hook_state.orig_cu_graph_kernel_node_set_params_v1.load(
			std::memory_order_acquire);
	this->original_cu_graph_kernel_node_set_params_v2 =
		hook_state.orig_cu_graph_kernel_node_set_params_v2.load(
			std::memory_order_acquire);
	this->original_cuda_memcpy_from_symbol =
		hook_state.orig_cuda_memcpy_from_symbol.load(
			std::memory_order_acquire);
	this->original_cuda_memcpy_from_symbol_async =
		hook_state.orig_cuda_memcpy_from_symbol_async.load(
			std::memory_order_acquire);

	static const char *ptx_pass_libraries = DEFAULT_PTX_PASS_EXECUTABLE;
	std::vector<std::filesystem::path> pass_libraries;
	{
		const char *provided_libraries =
			getenv("BPFTIME_PTXPASS_LIBRARIES");
		if (provided_libraries && strlen(provided_libraries) > 0) {
			ptx_pass_libraries = provided_libraries;
			SPDLOG_INFO(
				"Parsing user provided (by BPFTIME_PTXPASS_LIBRARIES) libraries: {}",
				provided_libraries);
		} else {
			SPDLOG_INFO("Parsing bundled libraries: {}",
				    ptx_pass_libraries);
		}
	}
	auto paths = split_by_colon(ptx_pass_libraries);
	for (const auto &path : paths) {
		SPDLOG_INFO("Found path: {}, executing..", path.c_str());
		void *handle = dlmopen(LM_ID_NEWLM, path.c_str(),
				       RTLD_NOW | RTLD_LOCAL);
		if (!handle) {
			SPDLOG_ERROR(
				"Unable to load dynamic library of pass {}: {}",
				path.c_str(), dlerror());
			continue;
		}
		auto print_config =
			(print_config_fn)dlsym(handle, "print_config");
		if (!print_config) {
			SPDLOG_ERROR("Symbol print_config not found in {}",
				     path.c_str());
			continue;
		}
		auto process_input =
			(process_input_fn)dlsym(handle, "process_input");
		if (!process_input) {
			SPDLOG_ERROR("Symbol process_input not found in {}",
				     path.c_str());
			continue;
		}
		ptxpass::pass_config::PassConfig config;
		std::vector<char> buf(10 << 20);
		print_config(buf.size(), buf.data());

		auto json = nlohmann::json::parse(buf.data());
		ptxpass::pass_config::from_json(json, config);
		SPDLOG_INFO("Retrived config of {}", path.c_str());
		SPDLOG_DEBUG("Config {}", json.dump(4));
		this->pass_configurations.emplace_back(
			std::make_unique<pass_cfg_with_exec_path>(
				path, config, print_config, process_input,
				handle));
	}
	{
		this->ptx_compiler = *load_nv_attach_impl_ptx_compiler(
			DEFAULT_PTX_COMPILER_SHARED_LIB,
			this->ptx_compiler_dl_handle);
	}
}

nv_attach_impl::~nv_attach_impl()
{
	if (nv_attach_get_active_impl() == this)
		nv_attach_set_active_impl(nullptr);

	if (frida_interceptor != nullptr) {
		auto interceptor = (GumInterceptor *)frida_interceptor;
		gum_interceptor_begin_transaction(interceptor);

		if (frida_listener != nullptr) {
			gum_interceptor_detach(
				interceptor,
				(GumInvocationListener *)frida_listener);
		}

		gum_interceptor_end_transaction(interceptor);
	}

	if (frida_listener) {
		g_object_unref(frida_listener);
		frida_listener = nullptr;
	}
	if (ptx_compiler_dl_handle) {
		dlclose(ptx_compiler_dl_handle);
	}
}

void nv_attach_impl::record_patched_kernel_function(
	const std::string &kernel_name, CUfunction function)
{
	if (kernel_name.empty() || function == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = patched_kernel_by_name.find(kernel_name);
	if (itr == patched_kernel_by_name.end()) {
		patched_kernel_by_name.emplace(kernel_name, function);
		return;
	}
	if (itr->second != function)
		itr->second = function;
}

std::optional<CUfunction> nv_attach_impl::find_patched_kernel_function(
	const std::string &kernel_name) const
{
	if (kernel_name.empty())
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = patched_kernel_by_name.find(kernel_name);
	if (itr == patched_kernel_by_name.end())
		return std::nullopt;
	return itr->second;
}

void nv_attach_impl::record_original_cufunction_name(
	CUfunction function, const std::string &kernel_name)
{
	if (function == nullptr || kernel_name.empty())
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cufunction.find(function);
	if (itr == kernel_name_by_cufunction.end()) {
		kernel_name_by_cufunction.emplace(function, kernel_name);
		return;
	}
	if (itr->second != kernel_name)
		itr->second = kernel_name;
}

std::optional<std::string>
nv_attach_impl::find_original_kernel_name(CUfunction function) const
{
	if (function == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cufunction.find(function);
	if (itr == kernel_name_by_cufunction.end())
		return std::nullopt;
	return itr->second;
}

std::map<std::string, std::string>
nv_attach_impl::extract_ptxs(std::vector<uint8_t> &&data_vec)
{
	std::map<std::string, std::string> all_ptx;
	if (const char *dir = getenv("BPFTIME_CUDA_LATE_PTX_DIR");
	    dir && dir[0] != '\0') {
		std::error_code ec;
		std::filesystem::path p(dir);
		if (std::filesystem::exists(p, ec) &&
		    std::filesystem::is_directory(p, ec)) {
			for (const auto &entry :
			     std::filesystem::directory_iterator(p, ec)) {
				if (ec)
					break;
				if (!entry.is_regular_file())
					continue;
				if (!entry.path().string().ends_with(".ptx"))
					continue;
				std::ifstream ifs(entry.path());
				if (!ifs.is_open())
					continue;
				std::stringstream buffer;
				buffer << ifs.rdbuf();
				all_ptx[entry.path().filename()] = buffer.str();
			}
			SPDLOG_INFO(
				"Using externally extracted PTX dir {} ({} file(s))",
				p.c_str(), all_ptx.size());
			return all_ptx;
		}
		SPDLOG_WARN(
			"BPFTIME_CUDA_LATE_PTX_DIR={} is not a directory; falling back to cuobjdump",
			dir);
	}

	if (const char *dis = getenv("BPFTIME_CUDA_DISABLE_CUOBJDUMP");
	    dis && dis[0] == '1') {
		SPDLOG_WARN(
			"Skipping cuobjdump-based PTX extraction because BPFTIME_CUDA_DISABLE_CUOBJDUMP=1");
		return all_ptx;
	}

	char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
	if (mkdtemp(tmp_dir) == nullptr) {
		SPDLOG_ERROR("mkdtemp failed for {}: {}", tmp_dir,
			     strerror(errno));
		return all_ptx;
	}
	auto working_dir = std::filesystem::path(tmp_dir);
	auto fatbin_path = working_dir / "temp.fatbin";
	{
		std::ofstream ofs(fatbin_path, std::ios::binary);
		ofs.write((const char *)data_vec.data(), data_vec.size());
		SPDLOG_INFO("Temporary fatbin written to {}",
			    fatbin_path.c_str());
	}
	SPDLOG_INFO("Extracting PTX in the fatbin...");

	const auto find_cuobjdump = []() -> std::string {
		const auto exists = [](const std::filesystem::path &p) -> bool {
			std::error_code ec;
			return std::filesystem::exists(p, ec);
		};

		if (const char *p = getenv("BPFTIME_CUOBJDUMP");
		    p && p[0] != '\0') {
			std::string s(p);
			if (s.find('/') == std::string::npos)
				return s;
			if (exists(s))
				return s;
		}

		const auto try_root =
			[&](const char *env) -> std::optional<std::string> {
			if (const char *p = getenv(env); p && p[0] != '\0') {
				auto cand = std::filesystem::path(p) / "bin" /
					    "cuobjdump";
				if (exists(cand))
					return cand.string();
			}
			return std::nullopt;
		};
		for (const char *env :
		     { "BPFTIME_CUDA_ROOT", "CUDA_HOME", "CUDA_PATH",
		       "LLVMBPF_CUDA_PATH", "CUDAToolkit_ROOT" }) {
			if (auto p = try_root(env))
				return *p;
		}

#if __linux__
		{
			std::error_code ec;
			const std::filesystem::path usr_local("/usr/local");
			for (const auto &entry :
			     std::filesystem::directory_iterator(usr_local,
								 ec)) {
				if (ec)
					break;
				if (!entry.is_directory())
					continue;
				auto name = entry.path().filename().string();
				if (name != "cuda" &&
				    !name.starts_with("cuda-"))
					continue;
				auto cand = entry.path() / "bin" / "cuobjdump";
				if (exists(cand))
					return cand.string();
			}
		}
#endif

		return "cuobjdump";
	};

	const auto cuobjdump = find_cuobjdump();
	SPDLOG_INFO("Calling cuobjdump: {} --extract-ptx all {}",
		    cuobjdump.c_str(), fatbin_path.c_str());

#if __linux__
	{
		const auto sh_quote = [](const std::string &s) -> std::string {
			std::string out;
			out.reserve(s.size() + 8);
			out.push_back('\'');
			for (char c : s) {
				if (c == '\'')
					out += "'\"'\"'";
				else
					out.push_back(c);
			}
			out.push_back('\'');
			return out;
		};

		std::string cmd;
		cmd.reserve(256);
		cmd += "cd -- ";
		cmd += sh_quote(std::string(tmp_dir));
		cmd += " && ";
		cmd += sh_quote(cuobjdump);
		cmd += " --extract-ptx all temp.fatbin";

		std::vector<std::string> arg_strs;
		arg_strs.emplace_back("sh");
		arg_strs.emplace_back("-c");
		arg_strs.emplace_back(std::move(cmd));
		std::vector<char *> argv;
		argv.reserve(arg_strs.size() + 1);
		for (auto &s : arg_strs)
			argv.push_back(s.data());
		argv.push_back(nullptr);

		std::vector<std::string> env_strs;
		for (char **p = ::environ; p && *p; ++p) {
			if (strncmp(*p, "LD_PRELOAD=", 11) == 0)
				continue;
			env_strs.emplace_back(*p);
		}
		env_strs.emplace_back("LD_PRELOAD=");
		std::vector<char *> envp;
		envp.reserve(env_strs.size() + 1);
		for (auto &s : env_strs)
			envp.push_back(s.data());
		envp.push_back(nullptr);

		pid_t child_pid = -1;
		int rc = posix_spawnp(&child_pid, "/bin/sh", nullptr, nullptr,
				      argv.data(), envp.data());
		if (rc != 0) {
			SPDLOG_ERROR("posix_spawnp(/bin/sh) failed: rc={}", rc);
			return all_ptx;
		}
		int status = 0;
		if (waitpid(child_pid, &status, 0) < 0) {
			SPDLOG_ERROR("waitpid(cuobjdump) failed: {}",
				     strerror(errno));
			return all_ptx;
		}
		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
			SPDLOG_ERROR("cuobjdump failed: status={}", status);
			return all_ptx;
		}
	}
#else
	{
		boost::process::ipstream stream;
		boost::process::environment env =
			boost::this_process::environment();
		env["LD_PRELOAD"] = "";

		auto cuobjdump_cmd_line = cuobjdump + " --extract-ptx all " +
					  fatbin_path.string();
		SPDLOG_INFO("Calling cuobjdump: {}", cuobjdump_cmd_line);

		boost::process::child child(
			"/bin/sh",
			boost::process::args({ "-c", cuobjdump_cmd_line }),
			boost::process::std_out > stream,
			boost::process::env(env),
			boost::process::start_dir = tmp_dir);

		std::string line;
		while (std::getline(stream, line)) {
			SPDLOG_DEBUG("cuobjdump output: {}", line);
		}
	}
#endif

	for (const auto &entry :
	     std::filesystem::directory_iterator(working_dir)) {
		if (entry.is_regular_file() &&
		    entry.path().string().ends_with(".ptx")) {
			std::ifstream ifs(entry.path());
			std::stringstream buffer;
			buffer << ifs.rdbuf();
			all_ptx[entry.path().filename()] = buffer.str();
		}
	}
	if (!spdlog::should_log(spdlog::level::debug)) {
		SPDLOG_INFO("Remove extracted files..");
		std::filesystem::remove_all(working_dir);
	}
	SPDLOG_INFO("Got {} PTX files", all_ptx.size());
	return all_ptx;
}
std::optional<std::map<std::string, std::tuple<std::string, bool>>>
nv_attach_impl::hack_fatbin(std::map<std::string, std::string> all_ptx)
{
	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::map<std::string, std::tuple<std::string, bool>> ptx_out;
	std::mutex map_mutex;
	std::mutex cache_mutex;
	for (auto &[file_name, original_ptx] : all_ptx) {
		boost::asio::post(pool, [this, original_ptx, file_name, &map_mutex, &ptx_out, &cache_mutex]() -> void {
			auto current_ptx = original_ptx;
			SPDLOG_INFO("Patching PTX: {}", file_name);
			bool should_add_trampoline = false;
			for (const auto &[_, hook_entry] : this->hook_entries) {
				const auto &kernels = hook_entry.kernels;
				for (const auto &kernel : kernels) {
					std::vector<uint64_t> ebpf_inst_words;
					ebpf_inst_words.assign(
						(uint64_t *)(uintptr_t)hook_entry
							.instuctions.data(),
						(uint64_t *)(uintptr_t)hook_entry
								.instuctions
								.data() +
							hook_entry.instuctions
								.size()

					);
					ptxpass::runtime_request::RuntimeRequest
						req;
					auto &ri = req.input;
					ri.full_ptx = current_ptx;
					ri.to_patch_kernel = kernel;
					ri.global_ebpf_map_info_symbol =
						"map_info";
					ri.ebpf_communication_data_symbol =
						"constData";

					req.set_ebpf_instructions(
						ebpf_inst_words);
					nlohmann::json in;
					ptxpass::runtime_request::to_json(in,
									  req);
					auto input_json = in.dump();
					SPDLOG_DEBUG("Input: {}", input_json);
					auto sha256_string =
						sha256(input_json.data(),
						       input_json.size());

					ptxpass::runtime_response::RuntimeResponse
						resp;

					cache_mutex.lock();
					if (auto itr = this->patch_cache.find(
						    sha256_string);
					    itr != this->patch_cache.end()) {
						SPDLOG_INFO(
							"Patching request {} found in cache",
							sha256_string);
						resp = itr->second;
						cache_mutex.unlock();
					} else {
						cache_mutex.unlock();
						SPDLOG_INFO(
							"Patching request {} not found in cache, patching..",
							sha256_string);
						bool parsed = false;
						std::string last_parse_error;
						constexpr size_t kMinBufBytes =
							1U << 20; // 1 MiB
						constexpr size_t kMaxBufBytes =
							64U << 20; // 64 MiB
						for (size_t buf_bytes =
							     kMinBufBytes;
						     buf_bytes <= kMaxBufBytes;
						     buf_bytes <<= 1) {
							try {
								std::vector<char> buf(
									buf_bytes);
								buf.back() =
									'\0';
								const int len =
									(buf_bytes >
									 (size_t)std::numeric_limits<
										 int>::
										 max()) ?
										std::numeric_limits<
											int>::
											max() :
										(int)buf_bytes;
								int err =
									hook_entry
										.config
										->process_input(
											input_json
												.c_str(),
											len,
											buf.data());
								if (err !=
								    ptxpass::ExitCode::
									    Success) {
									SPDLOG_ERROR(
										"Unable to run pass on kernel {}: {}",
										kernel,
										(int)err);
									return;
								}

								auto json = nlohmann::json::parse(
									buf.data(),
									nullptr,
									true,
									true);
								using namespace ptxpass::
									runtime_response;
								from_json(json,
									  resp);
								parsed = true;
								break;
							} catch (
								const std::exception
									&e) {
								last_parse_error =
									e.what();
							}
						}
						if (!parsed) {
							SPDLOG_ERROR(
								"Unable to parse PTX pass output for kernel {} (max {} MiB), last error: {}",
								kernel,
								(kMaxBufBytes >>
								 20),
								last_parse_error);
							return;
						}
						std::lock_guard<std::mutex>
							_cache_guard(
								cache_mutex);
						patch_cache[sha256_string] =
							resp;
					}
					current_ptx = resp.output_ptx;
					should_add_trampoline =
						should_add_trampoline ||
						resp.modified;
				}
			}
			if (should_add_trampoline) {
				current_ptx =
					ptxpass::filter_out_version_headers_ptx(
						wrap_ptx_with_trampoline(
							current_ptx));
			}
			std::lock_guard<std::mutex> _guard(map_mutex);
			ptx_out["patched." + file_name] = std::make_tuple(
				current_ptx, should_add_trampoline);
		});
	}
	pool.join();
	if (spdlog::should_log(spdlog::level::debug)) {
		char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
		if (mkdtemp(tmp_dir) == nullptr) {
			SPDLOG_ERROR("mkdtemp failed for {}: {}", tmp_dir,
				     strerror(errno));
			return std::nullopt;
		}
		auto working_dir = std::filesystem::path(tmp_dir);

		SPDLOG_DEBUG("Writing patched PTX to {}", working_dir.c_str());
		for (const auto &[file_name, ptx] : ptx_out) {
			auto path = working_dir / (file_name);
			std::ofstream ofs(path);
			ofs << std::get<0>(ptx);
		}
	}
	return ptx_out;
}

namespace bpftime::attach
{

int nv_attach_impl::find_attach_entry_by_program_name(const char *name) const
{
	for (const auto &entry : this->hook_entries) {
		if (entry.second.program_name == name)
			return entry.first;
	}
	return -1;
}
#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			SPDLOG_ERROR("{} failed with error code {}", #x,       \
				     (int)result);                             \
			return -1;                                             \
		}                                                              \
	} while (0)
#define CUDA_SAFE_CALL(x)                                                      \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			const char *msg;                                       \
			cuGetErrorName(result, &msg);                          \
			SPDLOG_ERROR("{} failed with error {}", #x, msg);      \
			return -1;                                             \
		}                                                              \
	} while (0)

int nv_attach_impl::run_attach_entry_on_gpu(int attach_id, int run_count,
					    int grid_dim_x, int grid_dim_y,
					    int grid_dim_z, int block_dim_x,
					    int block_dim_y, int block_dim_z)
{
	if (this->shared_mem_ptr == 0) {
		SPDLOG_ERROR(
			"shared_mem_ptr is not initialized; cannot run attach {} on GPU",
			attach_id);
		return -1;
	}
	if (run_count < 1) {
		SPDLOG_ERROR("run_count must be greater than 0");
		return -1;
	}
	std::vector<ebpf_inst> insts;
	if (auto itr = hook_entries.find(attach_id);
	    itr != hook_entries.end()) {
		insts = itr->second.instuctions;
	} else {
		SPDLOG_ERROR("Invalid attach id {}", attach_id);
		return -1;
	}
	SPDLOG_INFO("Running program on GPU");

	std::string sm_arch = get_gpu_sm_arch();
	SPDLOG_INFO("Using SM architecture: {}", sm_arch);

	std::vector<uint64_t> ebpf_words;
	for (const auto &insts : insts) {
		ebpf_words.push_back(*(uint64_t *)(uintptr_t)&insts);
	}
	auto ptx = ptxpass::filter_out_version_headers_ptx(
		wrap_ptx_with_trampoline_for_sm(
			filter_compiled_ptx_for_ebpf_program(
				ptxpass::compile_ebpf_to_ptx_from_words(
					ebpf_words, sm_arch.c_str(), "bpf_main",
					false, false),
				"bpf_main"),
			sm_arch));
	{
		const std::string to_replace = ".func bpf_main";

		auto bpf_main_pos = ptx.find(to_replace);
		if (bpf_main_pos == ptx.npos) {
			SPDLOG_ERROR("Cannot find '{}' in generated PTX code",
				     to_replace);
			return -1;
		}
		ptx = ptx.replace(bpf_main_pos, to_replace.size(),
				  ".visible .entry bpf_main");
	}
	if (spdlog::get_level() <= SPDLOG_LEVEL_DEBUG) {
		auto path = "/tmp/directly-run.ptx";

		std::ofstream ofs(path);
		ofs << ptx << std::endl;
		SPDLOG_DEBUG("Dumped directly run ptx to {}", path);
	}
	std::vector<char> output_elf;
	{
		unsigned int major_ver, minor_ver;
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetVersion(&major_ver, &minor_ver));
		SPDLOG_INFO("PTX compiler version {}.{}", major_ver, minor_ver);
		nvPTXCompilerHandle compiler = nullptr;
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(
			&compiler, (size_t)ptx.size(), ptx.c_str()));
		std::string gpu_name_opt = "--gpu-name=" + sm_arch;
		const char *compile_options[] = { gpu_name_opt.c_str(),
						  "--verbose" };
		auto status = nvPTXCompilerCompile(
			compiler, std::size(compile_options), compile_options);
		if (status != NVPTXCOMPILE_SUCCESS) {
			size_t error_size;

			NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(
				compiler, &error_size));

			if (error_size != 0) {
				std::string error_log(error_size + 1, '\0');
				NVPTXCOMPILER_SAFE_CALL(
					nvPTXCompilerGetErrorLog(
						compiler, error_log.data()));
				SPDLOG_ERROR("Unable to compile: {}",
					     error_log);
			}
			return -1;
		}
		size_t compiled_size;
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(
			compiler, &compiled_size));
		output_elf.resize(compiled_size);
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
			compiler, (void *)output_elf.data()));
		size_t info_size;
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetInfoLogSize(compiler, &info_size));
		std::string info_log(info_size + 1, '\0');
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetInfoLog(compiler, info_log.data()));
		SPDLOG_INFO("{}", info_log);
	}
	SPDLOG_INFO("Compiled program size: {}", output_elf.size());
	{
		CUdevice cuDevice;
		CUcontext context;
		CUmodule module;
		CUfunction kernel;
		CUDA_SAFE_CALL(cuInit(0));
		CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

		CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
		CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, output_elf.data(), 0,
						  0, 0));
		{
			CUdeviceptr ptr;
			size_t bytes;
			CUDA_SAFE_CALL(cuModuleGetGlobal(&ptr, &bytes, module,
							 "constData"));
			CUDA_SAFE_CALL(
				cuMemcpyHtoD(ptr, &this->shared_mem_ptr,
					     sizeof(this->shared_mem_ptr)));
			SPDLOG_INFO(
				"shared_mem_ptr copied: device ptr {:x}, device size {}",
				(uintptr_t)ptr, bytes);
		}
		{
			CUdeviceptr ptr;
			size_t bytes;
			CUDA_SAFE_CALL(cuModuleGetGlobal(&ptr, &bytes, module,
							 "map_info"));
			if (!this->map_basic_info.has_value()) {
				SPDLOG_ERROR(
					"map_basic_info is not set, cannot copy to device");
				return -1;
			}
			CUDA_SAFE_CALL(cuMemcpyHtoD(
				ptr, this->map_basic_info->data(),
				sizeof(this->map_basic_info->at(0)) *
					this->map_basic_info->size()));
			SPDLOG_INFO(
				"map_info copied: device ptr {:x}, device size {}",
				(uintptr_t)ptr, bytes);
		}
		CUDA_SAFE_CALL(
			cuModuleGetFunction(&kernel, module, "bpf_main"));
		for (int i = 1; i <= run_count; i++) {
			SPDLOG_INFO("Run {}", i);
			CUDA_SAFE_CALL(cuLaunchKernel(
				kernel, grid_dim_x, grid_dim_y, grid_dim_z,
				block_dim_x, block_dim_y, block_dim_z, 0,
				nullptr, nullptr, 0));
			CUDA_SAFE_CALL(cuCtxSynchronize());
		}
	}
	return 0;
}

void nv_attach_impl::mirror_cuda_memcpy_to_symbol(
	const void *symbol, const void *src, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream, bool async)
{
	if (!is_enabled())
		return;
	bootstrap_existing_fatbins_once();

	std::optional<variable_info> resolved_var;

	if (auto record_itr = symbol_address_to_fatbin.find((void *)symbol);
	    record_itr != symbol_address_to_fatbin.end()) {
		auto &record = *record_itr->second;
		auto var_itr =
			record.variable_addr_to_symbol.find((void *)symbol);
		if (var_itr != record.variable_addr_to_symbol.end()) {
			resolved_var = var_itr->second;
		}
	}

	if (!resolved_var) {
		auto name = resolve_host_function_symbol((void *)symbol);
		if (!name)
			return;
		{
			std::lock_guard<std::mutex> guard(
				patched_global_cache_mutex);
			auto it = patched_global_by_name.find(*name);
			if (it != patched_global_by_name.end()) {
				resolved_var = variable_info{
					.symbol_name = *name,
					.ptr = it->second.first,
					.size = it->second.second,
					.ptx = nullptr,
				};
			}
		}
		if (!resolved_var) {
			for (const auto &rec_uptr : fatbin_records) {
				auto *rec = rec_uptr.get();
				if (rec == nullptr)
					continue;
				for (const auto &ptx : rec->ptxs) {
					CUdeviceptr dptr;
					size_t sz;
					auto err = cuModuleGetGlobal(
						&dptr, &sz, ptx->module_ptr,
						name->c_str());
					if (err == CUDA_SUCCESS) {
						{
							std::lock_guard<
								std::mutex>
								guard(patched_global_cache_mutex);
							patched_global_by_name[*name] =
								std::make_pair(
									dptr,
									sz);
						}
						resolved_var = variable_info{
							.symbol_name = *name,
							.ptr = dptr,
							.size = sz,
							.ptx = nullptr,
						};
						break;
					}
				}
				if (resolved_var)
					break;
			}
		}
	}

	if (!resolved_var)
		return;

	auto &var_info = *resolved_var;
	if (offset >= var_info.size) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: offset {} exceeds size {} for symbol {}",
			offset, var_info.size, var_info.symbol_name);
		return;
	}
	size_t writable = var_info.size - offset;
	size_t bytes_to_copy = std::min(count, writable);
	if (bytes_to_copy == 0)
		return;
	if (bytes_to_copy != count) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: truncating copy for symbol {} (requested={}, allowed={})",
			var_info.symbol_name, count, bytes_to_copy);
	}
	CUdeviceptr dst = var_info.ptr + offset;
	CUstream cu_stream = reinterpret_cast<CUstream>(stream);
	CUresult status = CUDA_SUCCESS;

	auto copy_device_ptr = [](const void *ptr) -> CUdeviceptr {
		return static_cast<CUdeviceptr>(
			reinterpret_cast<uintptr_t>(ptr));
	};

	switch (kind) {
	case cudaMemcpyHostToDevice:
	case cudaMemcpyDefault:
		status = async ? cuMemcpyHtoDAsync(dst, src, bytes_to_copy,
						   cu_stream) :
				 cuMemcpyHtoD(dst, src, bytes_to_copy);
		break;
	case cudaMemcpyDeviceToDevice:
		status = async ? cuMemcpyDtoDAsync(dst, copy_device_ptr(src),
						   bytes_to_copy, cu_stream) :
				 cuMemcpyDtoD(dst, copy_device_ptr(src),
					      bytes_to_copy);
		break;
	default:
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_to_symbol: unsupported memcpy kind {} for symbol {}",
			(int)kind, var_info.symbol_name);
		return;
	}
	if (status != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: failed to copy symbol {} (err={})",
			var_info.symbol_name, (int)status);
	}
}

void nv_attach_impl::record_patched_launch(cudaStream_t stream)
{
	record_patched_launch_event(reinterpret_cast<CUstream>(stream));
}

void nv_attach_impl::record_patched_launch_event(CUstream stream)
{
	CUevent ev = nullptr;
	if (cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS ||
	    ev == nullptr) {
		return;
	}
	if (cuEventRecord(ev, stream) != CUDA_SUCCESS) {
		cuEventDestroy(ev);
		return;
	}
	{
		std::lock_guard<std::mutex> guard(launch_event_mutex);
		auto it = pending_launch_events_by_stream.find(stream);
		if (it != pending_launch_events_by_stream.end()) {
			if (it->second)
				cuEventDestroy(it->second);
			it->second = ev;
		} else {
			pending_launch_events_by_stream.emplace(stream, ev);
		}
	}
}

void nv_attach_impl::wait_for_patched_launch_events(
	std::chrono::milliseconds timeout)
{
	{
		CUcontext current = nullptr;
		if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
		    current == nullptr) {
			cuInit(0);
			CUdevice dev = 0;
			if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
				cuDevicePrimaryCtxRetain(&current, dev);
				if (current)
					cuCtxSetCurrent(current);
			}
		}
	}
	auto deadline = std::chrono::steady_clock::now() + timeout;
	for (;;) {
		std::vector<std::pair<CUstream, CUevent>> batch;
		{
			std::lock_guard<std::mutex> guard(launch_event_mutex);
			batch.reserve(pending_launch_events_by_stream.size());
			for (auto &kv : pending_launch_events_by_stream)
				batch.emplace_back(kv.first, kv.second);
			pending_launch_events_by_stream.clear();
		}
		if (batch.empty())
			return;
		std::vector<std::pair<CUstream, CUevent>> not_ready;
		not_ready.reserve(batch.size());
		for (auto &kv : batch) {
			CUstream stream = kv.first;
			CUevent ev = kv.second;
			if (ev == nullptr)
				continue;
			CUresult st = cuEventQuery(ev);
			if (st == CUDA_SUCCESS) {
				cuEventDestroy(ev);
			} else if (st == CUDA_ERROR_NOT_READY) {
				not_ready.emplace_back(stream, ev);
			} else {
				cuEventDestroy(ev);
			}
		}
		if (not_ready.empty())
			return;
		if (std::chrono::steady_clock::now() >= deadline) {
			std::lock_guard<std::mutex> guard(launch_event_mutex);
			for (auto &kv : not_ready) {
				auto it = pending_launch_events_by_stream.find(
					kv.first);
				if (it !=
				    pending_launch_events_by_stream.end()) {
					if (it->second)
						cuEventDestroy(it->second);
					it->second = kv.second;
				} else {
					pending_launch_events_by_stream.emplace(
						kv.first, kv.second);
				}
			}
			SPDLOG_WARN(
				"nv_attach_impl: detach timeout waiting for {} patched kernel event(s)",
				not_ready.size());
			return;
		}
		{
			std::lock_guard<std::mutex> guard(launch_event_mutex);
			for (auto &kv : not_ready) {
				auto it = pending_launch_events_by_stream.find(
					kv.first);
				if (it !=
				    pending_launch_events_by_stream.end()) {
					if (it->second)
						cuEventDestroy(it->second);
					it->second = kv.second;
				} else {
					pending_launch_events_by_stream.emplace(
						kv.first, kv.second);
				}
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
}

void nv_attach_impl::clear_patched_state_for_next_session()
{
	std::vector<std::unique_ptr<fatbin_record>> old_records;
	{
		std::lock_guard<std::mutex> guard(late_bootstrap_mutex);
		old_records.swap(fatbin_records);
		current_fatbin = nullptr;
		symbol_address_to_fatbin.clear();
		patch_cache.clear();
		{
			std::lock_guard<std::mutex> g(cuda_symbol_map_mutex);
			patched_kernel_by_name.clear();
			kernel_name_by_cufunction.clear();
		}
		{
			std::lock_guard<std::mutex> g(
				patched_global_cache_mutex);
			patched_global_by_name.clear();
		}
		if (module_pool)
			module_pool->clear();
		if (ptx_pool)
			ptx_pool->clear();
		{
			std::lock_guard<std::mutex> g(launch_event_mutex);
			for (auto &kv : pending_launch_events_by_stream) {
				if (kv.second)
					cuEventDestroy(kv.second);
			}
			pending_launch_events_by_stream.clear();
		}
	}
	old_records.clear();
}

bool nv_attach_impl::is_enabled() const noexcept
{
	return enabled.load(std::memory_order_acquire);
}

std::vector<std::string> nv_attach_impl::collect_all_kernels_to_patch() const
{
	std::vector<std::string> kernels;
	for (const auto &[_, entry] : hook_entries) {
		for (const auto &k : entry.kernels) {
			kernels.push_back(k);
		}
	}
	std::sort(kernels.begin(), kernels.end());
	kernels.erase(std::unique(kernels.begin(), kernels.end()),
		      kernels.end());
	return kernels;
}

void nv_attach_impl::build_host_symbol_cache_once()
{
	std::call_once(host_symbol_cache_once, [&]() {
		std::lock_guard<std::mutex> guard(host_symbol_cache_mutex);
		host_symbol_ranges.clear();

		auto modules = elf_introspect::list_loaded_modules();
		for (const auto &mod : modules) {
			auto syms = elf_introspect::read_function_symbols(mod);
			for (auto &sym : syms) {
				if (sym.name.empty() || sym.start == 0)
					continue;
				host_symbol_ranges.push_back(host_symbol_range{
					.start = sym.start,
					.end = sym.end,
					.name = std::move(sym.name) });
			}
		}
		std::sort(host_symbol_ranges.begin(), host_symbol_ranges.end(),
			  [](const auto &a, const auto &b) {
				  return a.start < b.start;
			  });
		if (!host_symbol_ranges.empty()) {
			SPDLOG_INFO("nv_attach_impl: cached {} host symbols",
				    host_symbol_ranges.size());
		}
	});
}

std::optional<std::string>
nv_attach_impl::resolve_host_function_symbol(void *addr)
{
	if (addr == nullptr)
		return std::nullopt;
	const auto normalize_cuda_stub = [](std::string s) -> std::string {
		constexpr const char *kStubPrefix = "__device_stub__";
		if (s.rfind(kStubPrefix, 0) == 0) {
			s.erase(0, strlen(kStubPrefix));
			if (!s.empty() && s[0] != '_')
				s.insert(s.begin(), '_');
		}
		return s;
	};

	Dl_info info{};
	if (dladdr(addr, &info) != 0 && info.dli_sname != nullptr &&
	    info.dli_sname[0] != '\0') {
		return normalize_cuda_stub(std::string(info.dli_sname));
	}

	build_host_symbol_cache_once();
	std::lock_guard<std::mutex> guard(host_symbol_cache_mutex);
	if (host_symbol_ranges.empty())
		return std::nullopt;

	const std::uintptr_t needle = (std::uintptr_t)addr;
	auto it = std::upper_bound(
		host_symbol_ranges.begin(), host_symbol_ranges.end(), needle,
		[](std::uintptr_t v, const auto &p) { return v < p.start; });
	if (it == host_symbol_ranges.begin())
		return std::nullopt;
	--it;
	if (it->end > it->start) {
		if (needle >= it->end)
			return std::nullopt;
	}
	return normalize_cuda_stub(it->name);
}

void nv_attach_impl::prefill_patched_kernel_functions_from_loaded_fatbins()
{
	auto kernels = collect_all_kernels_to_patch();
	if (kernels.empty())
		return;
	if (fatbin_records.empty())
		return;

	for (const auto &rec_uptr : fatbin_records) {
		auto *rec = rec_uptr.get();
		if (rec == nullptr)
			continue;
		for (const auto &ptx : rec->ptxs) {
			for (const auto &kernel : kernels) {
				CUfunction func = nullptr;
				auto err = cuModuleGetFunction(
					&func, ptx->module_ptr, kernel.c_str());
				if (err == CUDA_SUCCESS && func != nullptr) {
					record_patched_kernel_function(kernel,
								       func);
					record_original_cufunction_name(func,
									kernel);
				}
			}
		}
	}
}

namespace
{
struct __attribute__((__packed__)) fat_elf_header_t {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t size;
};

static void ensure_cuda_context_for_current_thread()
{
	CUcontext current = nullptr;
	CUresult err = cuCtxGetCurrent(&current);
	if (err == CUDA_SUCCESS && current != nullptr)
		return;

	err = cuInit(0);
	if (err != CUDA_SUCCESS)
		return;
	int dev_index = 0;
	if (const char *p = getenv("BPFTIME_CUDA_DEVICE"); p && p[0] != '\0') {
		try {
			dev_index = std::stoi(std::string(p));
			if (dev_index < 0)
				dev_index = 0;
		} catch (...) {
			dev_index = 0;
		}
	}
	CUdevice dev = 0;
	err = cuDeviceGet(&dev, dev_index);
	if (err != CUDA_SUCCESS)
		return;
	CUcontext ctx = nullptr;
	err = cuDevicePrimaryCtxRetain(&ctx, dev);
	if (err != CUDA_SUCCESS || ctx == nullptr)
		return;
	cuCtxSetCurrent(ctx);
}

static bool is_mapped_address(const void *p)
{
#if __linux__
	if (p == nullptr)
		return false;
	long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		return true;
	auto addr = (uintptr_t)p & ~(uintptr_t)(page_size - 1);
	unsigned char vec = 0;
	if (mincore((void *)addr, (size_t)page_size, &vec) == 0)
		return true;
	if (errno == ENOMEM)
		return false;
	return true;
#else
	(void)p;
	return true;
#endif
}

static std::optional<std::vector<uint8_t>>
read_fatbin_bytes_from_ptr(const void *ptr)
{
	if (ptr == nullptr)
		return std::nullopt;
	if (!is_mapped_address(ptr))
		return std::nullopt;
	const auto *data = reinterpret_cast<const char *>(ptr);
	auto *curr_header = (fat_elf_header_t *)data;
	const char *tail = reinterpret_cast<const char *>(curr_header);

	while (true) {
		if (!is_mapped_address(curr_header))
			break;
		if (curr_header->magic == 0xBA55ED50) {
			const char *next = ((const char *)curr_header) +
					   curr_header->header_size +
					   (std::size_t)curr_header->size;
			if (next <= tail)
				break;
			tail = next;
			curr_header = (fat_elf_header_t *)tail;
		} else {
			break;
		}
	}
	if (tail <= data)
		return std::nullopt;
	const std::size_t size = (std::size_t)(tail - data);
	constexpr std::size_t kMaxFatbinBytes = 512ull * 1024ull * 1024ull;
	if (size == 0 || size > kMaxFatbinBytes)
		return std::nullopt;
	return std::vector<uint8_t>((const uint8_t *)data,
				    (const uint8_t *)tail);
}

static std::optional<std::vector<uint8_t>>
read_fatbin_bytes_from_wrapper(const __fatBinC_Wrapper_t &wrapper)
{
	return read_fatbin_bytes_from_ptr(wrapper.data);
}
} // namespace

void nv_attach_impl::bootstrap_existing_fatbins()
{
	SPDLOG_INFO("nv_attach_impl: late attach bootstrap scanning fatbins..");
	auto modules = elf_introspect::list_loaded_modules();
	std::size_t ingested = 0;

	for (const auto &mod : modules) {
		if (auto sec = elf_introspect::find_section_in_memory(
			    mod, ".nv_fatbin");
		    sec) {
			auto sec_addr = sec->first;
			if (auto bytes = read_fatbin_bytes_from_ptr(sec_addr);
			    bytes) {
				auto extracted_ptx =
					extract_ptxs(std::move(*bytes));
				if (!extracted_ptx.empty()) {
					auto rec =
						std::make_unique<fatbin_record>();
					rec->original_ptx =
						std::move(extracted_ptx);
					rec->module_pool = module_pool;
					rec->ptx_pool = ptx_pool;
					rec->try_loading_ptxs(*this);
					fatbin_records.emplace_back(
						std::move(rec));
					ingested++;
					SPDLOG_INFO(
						"nv_attach_impl: ingested fatbin from .nv_fatbin in {}",
						mod.path.c_str());
				}
			}
		}

		if (auto sec = elf_introspect::find_section_in_memory(
			    mod, ".nvFatBinSegment");
		    sec) {
			auto [sec_addr, sec_size] = *sec;
			auto wrappers = elf_introspect::scan_fatbin_wrappers(
				sec_addr, sec_size);
			if (!wrappers.empty()) {
				SPDLOG_INFO(
					"nv_attach_impl: found {} fatbin wrapper(s) in {}",
					wrappers.size(), mod.path.c_str());
			}
			for (const auto *w : wrappers) {
				if (w == nullptr)
					continue;
				auto bytes = read_fatbin_bytes_from_wrapper(*w);
				if (!bytes)
					continue;
				auto extracted_ptx =
					extract_ptxs(std::move(*bytes));
				if (extracted_ptx.empty())
					continue;

				auto rec = std::make_unique<fatbin_record>();
				rec->original_ptx = std::move(extracted_ptx);
				rec->module_pool = module_pool;
				rec->ptx_pool = ptx_pool;
				rec->try_loading_ptxs(*this);
				fatbin_records.emplace_back(std::move(rec));
				ingested++;
			}
		}
	}

	if (ingested == 0) {
		auto extracted_ptx = extract_ptxs({});
		if (!extracted_ptx.empty()) {
			auto rec = std::make_unique<fatbin_record>();
			rec->original_ptx = std::move(extracted_ptx);
			rec->module_pool = module_pool;
			rec->ptx_pool = ptx_pool;
			rec->try_loading_ptxs(*this);
			fatbin_records.emplace_back(std::move(rec));
			ingested++;
			SPDLOG_INFO(
				"nv_attach_impl: ingested fatbin via external PTX directory fallback");
		}
	}

	SPDLOG_INFO(
		"nv_attach_impl: late attach bootstrap ingested {} fatbin(s)",
		ingested);
	prefill_patched_kernel_functions_from_loaded_fatbins();
}

void nv_attach_impl::bootstrap_existing_fatbins_once()
{
	if (!is_enabled())
		return;
	if (shared_mem_ptr == 0)
		return;
	if (!map_basic_info.has_value() || map_basic_info->empty())
		return;
	if (hook_entries.empty())
		return;

	std::lock_guard<std::mutex> guard(late_bootstrap_mutex);
	if (late_bootstrap_done.load(std::memory_order_acquire))
		return;
	if (!late_bootstrap_once) {
		late_bootstrap_once = std::make_unique<std::once_flag>();
	}
	std::call_once(*late_bootstrap_once, [&]() {
		try {
			ensure_cuda_context_for_current_thread();
			build_host_symbol_cache_once();
			bootstrap_existing_fatbins();
		} catch (const std::exception &ex) {
			SPDLOG_WARN("nv_attach_impl: late bootstrap failed: {}",
				    ex.what());
		}
	});
	late_bootstrap_done.store(true, std::memory_order_release);
}

void nv_attach_impl::reset_late_bootstrap_state_for_next_attach()
{
	std::lock_guard<std::mutex> guard(late_bootstrap_mutex);
	late_bootstrap_done.store(false, std::memory_order_release);
	late_bootstrap_started.store(false, std::memory_order_release);
	late_bootstrap_once = std::make_unique<std::once_flag>();
}

void nv_attach_impl::start_late_bootstrap_async()
{
	bool expected = false;
	if (!late_bootstrap_started.compare_exchange_strong(
		    expected, true, std::memory_order_acq_rel,
		    std::memory_order_acquire)) {
		return;
	}
	try {
		ensure_cuda_context_for_current_thread();
		this->bootstrap_existing_fatbins_once();
		if (!this->late_bootstrap_done.load(
			    std::memory_order_acquire)) {
			this->late_bootstrap_started.store(
				false, std::memory_order_release);
		}
	} catch (const std::exception &ex) {
		SPDLOG_WARN("nv_attach_impl: bootstrap failed: {}", ex.what());
		this->late_bootstrap_started.store(false,
						   std::memory_order_release);
	} catch (...) {
		SPDLOG_WARN("nv_attach_impl: bootstrap failed");
		this->late_bootstrap_started.store(false,
						   std::memory_order_release);
	}
}

} // namespace bpftime::attach
