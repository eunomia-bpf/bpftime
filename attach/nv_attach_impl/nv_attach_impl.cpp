#include "nv_attach_impl.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "ebpf_inst.h"
#include "frida-gum.h"

#include "nvPTXCompiler.h"
#include "nv_attach_private_data.hpp"
#include "nv_attach_utils.hpp"
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
#include <sstream>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <unistd.h>
#include <variant>
#include <vector>
#include <boost/asio.hpp>
#include <algorithm>
#include "ptxpass/core.hpp"
#include "ptx_pass_config.h"
using namespace bpftime;
using namespace attach;
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
	SPDLOG_INFO("Detaching is not supported by nv_attach_impl");
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

	// Safely access the variant
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
			break; // pass_definitions is sorted deterministically
			       // by executable
		}
	}
	if (matched) {
		auto id = this->allocate_id();
		nv_attach_entry entry;
		entry.instuctions = data.instructions;
		entry.kernels = data.func_names;
		entry.program_name = data.program_name;
		entry.config = matched;

		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		this->shared_mem_ptr = data.comm_shared_mem;
		SPDLOG_INFO("Recorded pass {} for func {}",
			    matched->executable_path.c_str(), func_name);
		return id;
	}
	// No matched definition: do not create generic entry; require explicit
	// pass definition to avoid ambiguous instrumentation.
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
}

nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
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
	{
		void *cuda_launch_kernel_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaLaunchKernel"));

		if (auto err = gum_interceptor_replace(
			    interceptor, cuda_launch_kernel_addr,
			    (gpointer)&cuda_runtime_function__cudaLaunchKernel,
			    this, nullptr);
		    err != GUM_REPLACE_OK) {
			SPDLOG_ERROR("Unable to replace cudaLaunchKernel: {}",
				     (int)err);
			assert(false);
		}
	}
	gum_interceptor_end_transaction(interceptor);

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
}

nv_attach_impl::~nv_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}
std::map<std::string, std::string>
nv_attach_impl::extract_ptxs(std::vector<uint8_t> &&data_vec)
{
	std::map<std::string, std::string> all_ptx;
	char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
	mkdtemp(tmp_dir);
	auto working_dir = std::filesystem::path(tmp_dir);
	auto fatbin_path = working_dir / "temp.fatbin";
	{
		std::ofstream ofs(fatbin_path, std::ios::binary);
		ofs.write((const char *)data_vec.data(), data_vec.size());
		SPDLOG_INFO("Temporary fatbin written to {}",
			    fatbin_path.c_str());
	}
	SPDLOG_INFO("Extracting PTX in the fatbin...");
	boost::asio::io_context ctx;
	boost::process::ipstream stream;
	boost::process::environment env = boost::this_process::environment();
	env["LD_PRELOAD"] = "";
	
	// Build command line - use shell to properly search PATH
	auto cuobjdump_cmd_line = std::string("cuobjdump --extract-ptx all ") +
				  fatbin_path.string();
	SPDLOG_INFO("Calling cuobjdump: {}", cuobjdump_cmd_line);
	
	// Execute through shell to properly use PATH
	boost::process::child child("/bin/sh",
				    boost::process::args({"-c", cuobjdump_cmd_line}),
				    boost::process::std_out > stream,
				    boost::process::env(env),
				    boost::process::start_dir = tmp_dir);

	std::string line;
	while (std::getline(stream, line)) {
		SPDLOG_DEBUG("cuobjdump output: {}", line);
	}
	for (const auto &entry :
	     std::filesystem::directory_iterator(working_dir)) {
		if (entry.is_regular_file() &&
		    entry.path().string().ends_with(".ptx")) {
			// Read the PTX into memory
			std::ifstream ifs(entry.path());
			std::stringstream buffer;
			buffer << ifs.rdbuf();
			all_ptx[entry.path().filename()] = buffer.str();
		}
	}
	SPDLOG_INFO("Got {} PTX files", all_ptx.size());
	return all_ptx;
}
std::optional<std::map<std::string, std::string>>
nv_attach_impl::hack_fatbin(std::map<std::string, std::string> all_ptx)
{
	char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
	mkdtemp(tmp_dir);
	auto working_dir = std::filesystem::path(tmp_dir);

	/**
	Here we can patch the PTX.
	*/
	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::map<std::string, std::string> ptx_out;
	std::mutex map_mutex;
	for (auto &[file_name, original_ptx] : all_ptx) {
		boost::asio::post(
			pool,
			[this, original_ptx, file_name, &map_mutex,
			 &ptx_out]() -> void {
				auto current_ptx = original_ptx;
				SPDLOG_INFO("Patching PTX: {}", file_name);

				for (const auto &[_, hook_entry] :
				     this->hook_entries) {
					const auto &kernels =
						hook_entry.kernels;
					for (const auto &kernel : kernels) {
						std::vector<uint64_t>
							ebpf_inst_words;
						ebpf_inst_words.assign(
							(uint64_t *)(uintptr_t)
								hook_entry
									.instuctions
									.data(),
							(uint64_t *)(uintptr_t)hook_entry
									.instuctions
									.data() +
								hook_entry
									.instuctions
									.size()

						);
						ptxpass::runtime_request::
							RuntimeRequest req;
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
						ptxpass::runtime_request::to_json(
							in, req);
						SPDLOG_DEBUG("Input: {}",
							     in.dump());
						std::vector<char> buf(200
								      << 20);
						int err =
							hook_entry.config->process_input(
								in.dump().c_str(),
								buf.size(),
								buf.data());
						if (err ==
						    ptxpass::ExitCode::Success) {
							auto json = nlohmann::json::
								parse(buf.data());
							using namespace ptxpass::
								runtime_response;
							RuntimeResponse resp;
							from_json(json, resp);
							current_ptx =
								resp.output_ptx;

						} else {
							SPDLOG_ERROR(
								"Unable to run pass on kernel {}: {}",
								kernel,
								(int)err);
						}
					}
				}
				current_ptx =
					ptxpass::filter_out_version_headers_ptx(
						wrap_ptx_with_trampoline(
							current_ptx));
				std::lock_guard<std::mutex> _guard(map_mutex);
				ptx_out["patched." + file_name] = current_ptx;
			});
	}
	pool.join();
	SPDLOG_INFO("Writting patched PTX to {}", working_dir.c_str());
	for (const auto &[file_name, ptx] : ptx_out) {
		auto path = working_dir / (file_name);
		std::ofstream ofs(path);
		ofs << ptx;
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
	if (run_count < 1) {
		SPDLOG_ERROR("run_count must be greater than 0");
		return -1;
	}
	std::vector<ebpf_inst> insts;
	if (auto itr = hook_entries.find(attach_id);
	    itr != hook_entries.end()) {
		// In new flow, directly_run is not supported and should be
		// represented by a dedicated pass
		insts = itr->second.instuctions;
	} else {
		SPDLOG_ERROR("Invalid attach id {}", attach_id);
		return -1;
	}
	SPDLOG_INFO("Running program on GPU");

	// Get SM architecture from environment variable, default to sm_60
	const char *sm_arch_env = std::getenv("BPFTIME_SM_ARCH");
	std::string sm_arch = sm_arch_env ? sm_arch_env : "sm_60";
	SPDLOG_INFO("Using SM architecture: {}", sm_arch);

	std::vector<uint64_t> ebpf_words;
	for (const auto &insts : insts) {
		ebpf_words.push_back(*(uint64_t *)(uintptr_t)&insts);
	}
	auto ptx = ptxpass::filter_out_version_headers_ptx(
		wrap_ptx_with_trampoline(filter_compiled_ptx_for_ebpf_program(
			ptxpass::compile_ebpf_to_ptx_from_words(
				ebpf_words, sm_arch.c_str(), "bpf_main", false, false),
			"bpf_main")));
	{
		const std::string to_replace = ".func bpf_main";

		// Replace ".func bpf_main" to ".visible .entry bpf_main" so it
		// can be executed
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
	// Compile to ELF
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
	// Load and run the program
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
		// fill data into it
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

void nv_attach_impl::rebase_gpu_ringbuf_map_buffers()
{
	SPDLOG_DEBUG(
		"nv_attach_impl::rebase_gpu_ringbuf_map_buffers is a no-op (legacy placeholder)");
}

void nv_attach_impl::mirror_cuda_memcpy_to_symbol(
	const void *symbol, const void *src, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream, bool async)
{
	auto record_itr = symbol_address_to_fatbin.find((void *)symbol);
	if (record_itr == symbol_address_to_fatbin.end())
		return;
	auto &record = *record_itr->second;
	auto var_itr = record.variable_addr_to_symbol.find((void *)symbol);
	if (var_itr == record.variable_addr_to_symbol.end()) {
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_to_symbol: no variable info for symbol pointer {:x}",
			(uintptr_t)symbol);
		return;
	}
	auto &var_info = var_itr->second;
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

} // namespace bpftime::attach
