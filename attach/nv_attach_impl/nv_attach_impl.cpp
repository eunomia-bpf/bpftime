#include "nv_attach_impl.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "ebpf_inst.h"
#include "frida-gum.h"

#include "llvm_jit_context.hpp"
#include "nvPTXCompiler.h"
#include "nv_attach_private_data.hpp"
#include "nv_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/process.hpp>
#include <chrono>
#include <cstdlib>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <thread>
#include <unistd.h>
#include <variant>
#include <vector>
#include <boost/asio.hpp>
using namespace bpftime;
using namespace attach;

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

	if (attach_type == ATTACH_CUDA_PROBE) {
		if (func_name == "__memcapture") {
			SPDLOG_INFO(
				"Recording memcapture in nv_attach_impl, instructions count = {}",
				data.instructions.size());
			auto id = this->allocate_id();
			hook_entries[id] = nv_attach_entry{
				.type = nv_attach_cuda_memcapture{},
				.instuctions = data.instructions,
				.kernels = data.func_names,
				.program_name = data.program_name
			};
			this->map_basic_info = data.map_basic_info;
			this->shared_mem_ptr = data.comm_shared_mem;
			return id;
		} else if (func_name == "__directly_run") {
			SPDLOG_INFO(
				"Recording directly run program on GPU, instructions count = {}",
				data.instructions.size());
			auto id = allocate_id();
			hook_entries[id] = nv_attach_entry{
				.type = nv_attach_directly_run_on_gpu{},
				.instuctions = data.instructions,
				.kernels = data.func_names,
				.program_name = data.program_name
			};
			this->map_basic_info = data.map_basic_info;
			this->shared_mem_ptr = data.comm_shared_mem;
		} else {
			SPDLOG_INFO("Recording kprobe for {}", func_name);
			auto id = this->allocate_id();
			hook_entries[id] = nv_attach_entry{
				.type =
					nv_attach_function_probe{
						.func = func_name,
						.is_retprobe = false,
					},
				.instuctions = data.instructions,
				.kernels = data.func_names,
				.program_name = data.program_name
			};
			this->map_basic_info = data.map_basic_info;
			this->shared_mem_ptr = data.comm_shared_mem;
			return id;
		}
	} else if (attach_type == ATTACH_CUDA_RETPROBE) {
		SPDLOG_INFO("Recording kretprobe for {}", func_name);
		auto id = this->allocate_id();
		hook_entries[id] =
			nv_attach_entry{ .type =
						 nv_attach_function_probe{
							 .func = func_name,
							 .is_retprobe = true,
						 },
					 .instuctions = data.instructions,
					 .kernels = data.func_names,
					 .program_name = data.program_name };
		this->map_basic_info = data.map_basic_info;
		this->shared_mem_ptr = data.comm_shared_mem;
		return id;
	} else {
		SPDLOG_ERROR("Unsupported attach type for nv_attach_impl: {}",
			     attach_type);
		return -1;
	}
	return 0;
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
			SPDLOG_ERROR("Unable to attach to CUDA functions: {}",
				     (int)result);
			throw std::runtime_error(
				"Failed to attach to CUDA function");
		}
	};

	void *register_fatbin_addr =
		dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
	register_hook(AttachedToFunction::RegisterFatbin, register_fatbin_addr);

	void *register_function_addr =
		GSIZE_TO_POINTER(gum_module_find_export_by_name(
			nullptr, "__cudaRegisterFunction"));
	register_hook(AttachedToFunction::RegisterFunction,
		      register_function_addr);

	void *register_variable_addr = GSIZE_TO_POINTER(
		gum_module_find_export_by_name(nullptr, "__cudaRegisterVar"));
	register_hook(AttachedToFunction::RegisterVariable,
		      register_variable_addr);

	void *register_fatbin_end_addr =
		GSIZE_TO_POINTER(gum_module_find_export_by_name(
			nullptr, "__cudaRegisterFatBinaryEnd"));
	register_hook(AttachedToFunction::RegisterFatbinEnd,
		      register_fatbin_end_addr);

	void *cuda_launch_kernel_addr = GSIZE_TO_POINTER(
		gum_module_find_export_by_name(nullptr, "cudaLaunchKernel"));

	if (auto err = gum_interceptor_replace(
		    interceptor, cuda_launch_kernel_addr,
		    (gpointer)&cuda_runtime_function__cudaLaunchKernel, this,
		    nullptr);
	    err != GUM_REPLACE_OK) {
		SPDLOG_ERROR("Unable to replace cudaLaunchKernel: {}",
			     (int)err);
		assert(false);
	}
	gum_interceptor_end_transaction(interceptor);
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
	auto cuobjdump_cmd_line = std::string("cuobjdump --extract-ptx all ") +
				  fatbin_path.string();
	SPDLOG_INFO("Calling cuobjdump: {}", cuobjdump_cmd_line);
	boost::process::child child(cuobjdump_cmd_line,
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
	std::mutex map_mutex;
	std::map<std::string, std::string> ptx_out;
	SPDLOG_INFO("Using thread pool to patch PTXs");
	for (auto &[file_name, original_ptx] : all_ptx) {
		boost::asio::post(pool, [file_name, original_ptx, this, &map_mutex, &ptx_out]() -> int {
			auto current_ptx = original_ptx;
			SPDLOG_INFO("Patching PTX: {}", file_name);
			bool trampoline_added = false;
			current_ptx = filter_unprintable_chars(current_ptx);
			for (const auto &[_, entry] : hook_entries) {
				if (std::holds_alternative<
					    nv_attach_cuda_memcapture>(
					    entry.type)) {
					SPDLOG_INFO(
						"Patching with memcapture..");
					if (auto result =
						    this->patch_with_memcapture(
							    current_ptx, entry,
							    !trampoline_added);
					    result.has_value()) {
						current_ptx = *result;
						trampoline_added = true;
					} else {
						SPDLOG_ERROR(
							"Failed to patch for memcapture");
						return -1;
					}
				} else if (std::holds_alternative<
						   nv_attach_function_probe>(
						   entry.type)) {
					SPDLOG_INFO(
						"Patching with kprobe/kretprobe");
					if (auto result =
						    this->patch_with_probe_and_retprobe(
							    current_ptx, entry,
							    !trampoline_added);
					    result.has_value()) {
						current_ptx = *result;
						trampoline_added = true;
					} else {
						SPDLOG_ERROR(
							"Failed to patch for probe/retprobe");
						return -1;
					}
				} else if (std::holds_alternative<
						   nv_attach_directly_run_on_gpu>(
						   entry.type)) {
					SPDLOG_INFO(
						"Found attach with nv_type nv_attach_directly_run_on_gpu, no need to patch");
				}
			}
			current_ptx = wrap_ptx_with_trampoline(current_ptx);
			current_ptx = filter_out_version_headers(current_ptx);
			current_ptx =
				add_semicolon_for_variable_lines(current_ptx);
			{
				// filter out comment lines
				std::istringstream iss(current_ptx);
				std::ostringstream oss;
				std::string line;
				while (std::getline(iss, line)) {
					if (line.starts_with("/"))
						continue;
					oss << line << std::endl;
				}
				std::lock_guard<std::mutex> _guard(map_mutex);
				ptx_out["patched." + file_name] = oss.str();
			}

			return 0;
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
std::string filter_unprintable_chars(std::string input)
{
	static const char non_printable_chars[] = {
		'\0',	'\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\a',
		'\b',	'\v',	'\f',	'\r',	'\x0E', '\x0F', '\x10', '\x11',
		'\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19',
		'\x1A', '\x1B', '\x1C', '\x1D', '\x1E', '\x1F', '\x7F'
	};
	std::set<char> set(std::begin(non_printable_chars),
			   std::end(non_printable_chars));

	std::string result;
	for (auto c : input) {
		if (set.contains(c) || (uint8_t)c > 127)
			continue;
		result.push_back(c);
	}
	while (!result.empty() && result.back() != '}')
		result.pop_back();
	if (result.empty()) {
		SPDLOG_ERROR(
			"filter_unprintable_chars: result is empty or no closing brace found");
		return "";
	}
	return result;
};
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
		if (std::holds_alternative<nv_attach_directly_run_on_gpu>(
			    itr->second.type)) {
			insts = itr->second.instuctions;

		} else {
			SPDLOG_ERROR(
				"Attach id {} is not expected to directly run on GPU",
				attach_id);
			return -1;
		}
	} else {
		SPDLOG_ERROR("Invalid attach id {}", attach_id);
		return -1;
	}
	SPDLOG_INFO("Running program on GPU");

	auto ptx = filter_out_version_headers(
		wrap_ptx_with_trampoline(filter_compiled_ptx_for_ebpf_program(
			generate_ptx_for_ebpf(insts, "bpf_main", false, false),
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
		const char *compile_options[] = { "--gpu-name=sm_60",
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

} // namespace bpftime::attach
