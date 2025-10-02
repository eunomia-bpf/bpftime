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
#include <boost/process/detail/child_decl.hpp>
#include <boost/process/env.hpp>
#include <boost/process/io.hpp>
#include <boost/process/pipe.hpp>
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
#include <memory>
#include <set>
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
#include <boost/process.hpp>
using namespace bpftime;
using namespace attach;

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
	if (attach_type == ATTACH_CUDA_PROBE) {
		if (std::get<std::string>(data.code_addr_or_func_name) ==
		    "__memcapture") {
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
		} else if (std::get<std::string>(data.code_addr_or_func_name) ==
			   "__directly_run") {
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
			SPDLOG_INFO("Recording kprobe for {}",
				    std::get<std::string>(
					    data.code_addr_or_func_name));
			auto id = this->allocate_id();
			hook_entries[id] = nv_attach_entry{
				.type =
					nv_attach_function_probe{
						.func = std::get<std::string>(
							data.code_addr_or_func_name),
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
		SPDLOG_INFO("Recording kretprobe for {}",
			    std::get<std::string>(data.code_addr_or_func_name));
		auto id = this->allocate_id();
		hook_entries[id] = nv_attach_entry{
			.type =
				nv_attach_function_probe{
					.func = std::get<std::string>(
						data.code_addr_or_func_name),
					.is_retprobe = true,
				},
			.instuctions = data.instructions,
			.kernels = data.func_names,
			.program_name = data.program_name
		};
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
nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	assert(interceptor != nullptr);
	auto listener =
		g_object_new(cuda_runtime_function_hooker_get_type(), nullptr);
	assert(listener != nullptr);
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;
	gum_interceptor_begin_transaction(interceptor);

	auto register_hook = [&](AttachedToFunction func, void *addr) {
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
			assert(false);
		}
	};
	register_hook(AttachedToFunction::RegisterFatbin,
		      (gpointer)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary"));

	register_hook(AttachedToFunction::RegisterFunction,
		      GSIZE_TO_POINTER(gum_module_find_export_by_name(
			      nullptr, "__cudaRegisterFunction")));
	register_hook(AttachedToFunction::RegisterFatbinEnd,
		      GSIZE_TO_POINTER(gum_module_find_export_by_name(
			      nullptr, "__cudaRegisterFatBinaryEnd")));
	register_hook(AttachedToFunction::CudaLaunchKernel,
		      GSIZE_TO_POINTER(gum_module_find_export_by_name(
			      nullptr, "cudaLaunchKernel")));

	gum_interceptor_end_transaction(interceptor);
}

nv_attach_impl::~nv_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}

std::optional<std::vector<uint8_t>>
nv_attach_impl::hack_fatbin(std::vector<uint8_t> &&data_vec)
{
	std::vector<std::string> ptx_out;
	{
		char tmp_dir[] = "/tmp/bpftime-fatbin-store.XXXXXX";
		mkdtemp(tmp_dir);
		auto output_path =
			std::filesystem::path(tmp_dir) / "temp.fatbin";
		{
			std::ofstream ofs(output_path, std::ios::binary);
			ofs.write((const char *)data_vec.data(),
				  data_vec.size());
			SPDLOG_INFO("Temporary fatbin written to {}",
				    output_path.c_str());
		}
		SPDLOG_INFO("Listing functions in the patched ptx");
		boost::asio::io_context ctx;
		boost::process::ipstream stream;
		boost::process::environment env =
			boost::this_process::environment();
		env["LD_PRELOAD"] = "";

		boost::process::child child(
			std::string("cuobjdump --dump-ptx ") +
				output_path.string(),
			boost::process::std_out > stream,
			boost::process::env(env));
		std::string line;
		std::string output;
		bool should_record = false;
		while (stream && std::getline(stream, line)) {
			if (should_record) {
				output += line + "\n";
			}
			if (line.starts_with("ptxasOptions = "))
				should_record = true;
		}
		ptx_out.push_back(output);
		{
			auto temp_out =
				std::filesystem::path(tmp_dir) / "temp.ptx";
			std::ofstream export_ofs(temp_out);
			export_ofs << output;
			SPDLOG_INFO("Extracted PTX at {}", temp_out.c_str());
		}
	}
	if (ptx_out.size() != 1) {
		SPDLOG_ERROR(
			"Expect the loaded fatbin to contain only 1 PTX code section, but it contains {}",
			ptx_out.size());
		return {};
	}

	/**
	Here we can patch the PTX. Then recompile it.
	*/
	bool trampoline_added = false;
	auto &to_patch_ptx = ptx_out[0];
	to_patch_ptx = filter_unprintable_chars(to_patch_ptx);
	for (const auto &[_, entry] : hook_entries) {
		if (std::holds_alternative<nv_attach_cuda_memcapture>(
			    entry.type)) {
			SPDLOG_INFO("Patching with memcapture..");
			if (auto result = this->patch_with_memcapture(
				    to_patch_ptx, entry, !trampoline_added);
			    result.has_value()) {
				to_patch_ptx = *result;
				trampoline_added = true;
			} else {
				SPDLOG_ERROR("Failed to patch for memcapture");
				return {};
			}
		} else if (std::holds_alternative<nv_attach_function_probe>(
				   entry.type)) {
			SPDLOG_INFO("Patching with kprobe/kretprobe");
			if (auto result = this->patch_with_probe_and_retprobe(
				    to_patch_ptx, entry, !trampoline_added);
			    result.has_value()) {
				to_patch_ptx = *result;
				trampoline_added = true;
			} else {
				SPDLOG_ERROR(
					"Failed to patch for probe/retprobe");
				return {};
			}
		} else if (std::holds_alternative<nv_attach_directly_run_on_gpu>(
				   entry.type)) {
			SPDLOG_INFO(
				"Found attach with nv_type nv_attach_directly_run_on_gpu, no need to patch");
		}
	}
	to_patch_ptx = wrap_ptx_with_trampoline(to_patch_ptx);
	to_patch_ptx = filter_out_version_headers(to_patch_ptx);
	{
		// filter out comment lines
		std::istringstream iss(to_patch_ptx);
		std::ostringstream oss;
		std::string line;
		while (std::getline(iss, line)) {
			if (line.starts_with("/"))
				continue;
			oss << line << std::endl;
		}
		to_patch_ptx = oss.str();
	}
	SPDLOG_INFO("Recompiling PTX with nvcc..");
	char tmp_dir[] = "/tmp/bpftime-recompile-nvcc";
	// mkdtemp(tmp_dir);
	std::filesystem::path work_dir(tmp_dir);
	if (!std::filesystem::exists(work_dir)) {
		SPDLOG_INFO("Creating work dir: {}", work_dir.c_str());
		std::filesystem::create_directories(work_dir);
	}
	SPDLOG_INFO("Working directory: {}", work_dir.c_str());
	// Detect proper CUDA arch for current device, fallback to env or sm_60
	std::string arch_flag;
	std::string code_flag;
	const char *env_arch = std::getenv("BPFTIME_CUDA_ARCH");
	std::string majmin;
	if (env_arch && std::strlen(env_arch) > 0) {
		// Normalize env arch
		if (std::strncmp(env_arch, "sm_", 3) == 0 &&
		    std::strlen(env_arch) >= 4) {
			majmin = std::string(env_arch + 3);
			arch_flag = std::string("-arch=compute_") + majmin;
			code_flag = std::string("-code=sm_") + majmin +
				    std::string(",compute_") + majmin;
		} else if (std::strncmp(env_arch, "compute_", 8) == 0 &&
			   std::strlen(env_arch) >= 9) {
			majmin = std::string(env_arch + 8);
			arch_flag =
				std::string("-arch=") + env_arch; // compute_XX
			code_flag = std::string("-code=sm_") + majmin +
				    std::string(",compute_") + majmin;
		} else {
			// Fallback if an unexpected value was provided
			majmin = "60";
			arch_flag = "-arch=compute_60";
			code_flag = "-code=sm_60,compute_60";
		}
	} else {
		int current_device = 0;
		cudaDeviceProp prop{};
		if (cudaGetDevice(&current_device) == cudaSuccess &&
		    cudaGetDeviceProperties(&prop, current_device) ==
			    cudaSuccess &&
		    prop.major >= 3) {
			majmin = std::to_string(prop.major) +
				 std::to_string(prop.minor);
			arch_flag = std::string("-arch=compute_") + majmin;
			code_flag = std::string("-code=sm_") + majmin +
				    std::string(",compute_") + majmin;
		} else {
			majmin = "60";
			arch_flag = "-arch=compute_60";
			code_flag = "-code=sm_60,compute_60";
		}
	}
	std::string command =
		std::string("nvcc -O2 -G -g --keep-device-functions ") +
		arch_flag +
		(code_flag.empty() ? std::string(" ") :
				     std::string(" ") + code_flag + " ");
	{
		auto ptx_in = work_dir / "main.ptx";
		// SPDLOG_WARN("Using /tmp/main.ptx as ptx for nvcc");
		// std::string ptx_in = "/tmp/main.ptx";
		SPDLOG_INFO("PTX IN: {}", ptx_in.c_str());
		std::ofstream ofs(ptx_in);
		// Ensure PTX has a valid header
		// (.version/.target/.address_size) Our earlier filter removed
		// headers; here we re-add a compatible header
		ofs << ".version 8.0\n.target sm_" << majmin
		    << "\n.address_size 64\n";
		// Fallback: ensure d_N exists for this benchmark if missing
		if (to_patch_ptx.find(" d_N") == std::string::npos &&
		    to_patch_ptx.find("\nd_N") == std::string::npos) {
			ofs << ".visible .const .align 4 .s32 d_N;\n";
		}
		// Write the patched PTX that includes trampoline symbols
		// (constData, map_info)
		ofs << to_patch_ptx;
		command += ptx_in;
		command += " ";
	}
	command += "-fatbin ";
	auto fatbin_out = work_dir / "out.fatbin";
	command += "-o ";
	command += fatbin_out;
	SPDLOG_INFO("Fatbin out {}", fatbin_out.c_str());
	SPDLOG_INFO("Starting nvcc: {}", command);
	if (int err = system(command.c_str()); err != 0) {
		SPDLOG_ERROR("Unable to execute nvcc");
		return {};
	}
	SPDLOG_INFO("NVCC execution done.");
	std::vector<uint8_t> fatbin_out_buf;
	{
		std::ifstream ifs(fatbin_out, std::ios::binary | std::ios::ate);
		auto file_tail = ifs.tellg();
		ifs.seekg(0, std::ios::beg);

		fatbin_out_buf.resize(file_tail);
		ifs.read((char *)fatbin_out_buf.data(), file_tail);
	}

	SPDLOG_INFO("Got patched fatbin in {} bytes", fatbin_out_buf.size());
	return fatbin_out_buf;
}

static uint64_t _constData_mock;
static char map_basic_info_mock[4096];
extern "C" {
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		       char *deviceAddress, const char *deviceName, int ext,
		       size_t size, int constant, int global);
}
int nv_attach_impl::register_trampoline_memory(void **fatbin_handle)
{
	if (this->trampoline_memory_state !=
	    TrampolineMemorySetupStage::NotSet) {
		SPDLOG_INFO("Invalid stage for register trampoline memory");
		return -1;
	}
	SPDLOG_INFO("Registering trampoline memory");

	__cudaRegisterVar(fatbin_handle, (char *)&_constData_mock,
			  (char *)nullptr, "constData", 0,
			  sizeof(_constData_mock), 1, 0);

	__cudaRegisterVar(fatbin_handle, (char *)&map_basic_info_mock,
			  (char *)nullptr, "map_info", 0,
			  sizeof(map_basic_info_mock), 1, 0);
	this->trampoline_memory_state = TrampolineMemorySetupStage::Registered;
	SPDLOG_INFO("Register trampoline memory done");
	return 0;
}
int nv_attach_impl::copy_data_to_trampoline_memory()
{
	SPDLOG_INFO("Copying data to device symbols..");
	const char *skip = std::getenv("BPFTIME_CUDA_SKIP_CONSTCOPY");
	if (skip && std::strlen(skip) > 0 &&
	    (std::string(skip) == "1" || std::string(skip) == "true")) {
		SPDLOG_INFO(
			"Skipping cudaMemcpyToSymbol due to BPFTIME_CUDA_SKIP_CONSTCOPY env");
		this->trampoline_memory_state =
			TrampolineMemorySetupStage::Copied;
		return 0;
	}
	if (auto err = cudaMemcpyToSymbol((const void *)&_constData_mock,
					  &this->shared_mem_ptr,
					  sizeof(_constData_mock));
	    err != cudaSuccess) {
		SPDLOG_ERROR(
			"Unable to copy `constData` (shared memory address) to device: {}",
			(int)err);
		return -1;
	}
	SPDLOG_INFO("Copying the followed map info:");
	for (int i = 0; i < this->map_basic_info->size(); i++) {
		const auto &cur = this->map_basic_info->at(i);
		if (cur.enabled) {
			SPDLOG_INFO(
				"Mapid {}, enabled = {}, key_size = {}, value_size = {}, max_ent={}, type={}",
				i, cur.enabled, cur.key_size, cur.value_size,
				cur.max_entries, cur.map_type);
		}
	}
	if (auto err = cudaMemcpyToSymbol((const void *)&map_basic_info_mock,
					  this->map_basic_info->data(),
					  sizeof(map_basic_info_mock));
	    err != cudaSuccess) {
		SPDLOG_ERROR("Unable to copy `map_basic_into` to device : {}",
			     (int)err);
		return -1;
	}
	this->trampoline_memory_state = TrampolineMemorySetupStage::Copied;
	SPDLOG_INFO("constData and map_basic_info copied..");

	return 0;
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
	result.reserve(input.size());
	for (auto c : input) {
		if (set.contains(c) || (uint8_t)c > 127)
			continue;
		result.push_back(c);
	}
	// Keep full PTX text; do not trim by last brace to avoid dropping
	// globals
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
		assert(bpf_main_pos != ptx.npos);
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
