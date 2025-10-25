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
#include <memory>
#include <set>
#include <regex>
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

	// New: select pass by scanning configurations (type + regex)
	const PassDefinition *matched = nullptr;
	std::regex func_re;
	for (const auto &pd : this->pass_definitions) {
		if (pd.attach_point.type != attach_type)
			continue;
		try {
			func_re = std::regex(
				pd.attach_point.expected_func_name_regex);
		} catch (...) {
			continue;
		}
		if (std::regex_match(func_name, func_re)) {
			matched = &pd;
			break; // pass_definitions is sorted by priority then
			       // name
		}
	}
	if (matched) {
		auto id = this->allocate_id();
		nv_attach_entry entry;
		entry.instuctions = data.instructions;
		entry.kernels = data.func_names;
		entry.program_name = data.program_name;
		entry.pass_exec = matched->executable;
		entry.priority = matched->priority;
		entry.parameters["func_name"] = func_name;
		entry.parameters["attach_type"] = std::to_string(attach_type);
		// derive attach point override for fallback pipeline
		if (attach_type == ATTACH_CUDA_PROBE) {
			if (func_name == "__memcapture")
				entry.attach_point_override =
					std::string("kprobe/__memcapture");
			else
				entry.attach_point_override =
					std::string("kprobe/") + func_name;
		} else if (attach_type == ATTACH_CUDA_RETPROBE) {
			entry.attach_point_override =
				std::string("kretprobe/") + func_name;
		}
		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		this->shared_mem_ptr = data.comm_shared_mem;
		SPDLOG_INFO("Recorded pass {} (priority {}) for func {}",
			    matched->executable, matched->priority, func_name);
		return id;
	}
	// No matched definition: still record a generic entry for fallback
	// pipeline
	{
		auto id = this->allocate_id();
		nv_attach_entry entry;
		entry.instuctions = data.instructions;
		entry.kernels = data.func_names;
		entry.program_name = data.program_name;
		// derive attach point override
		if (attach_type == ATTACH_CUDA_PROBE) {
			if (func_name == "__memcapture")
				entry.attach_point_override =
					std::string("kprobe/__memcapture");
			else
				entry.attach_point_override =
					std::string("kprobe/") + func_name;
		} else if (attach_type == ATTACH_CUDA_RETPROBE) {
			entry.attach_point_override =
				std::string("kretprobe/") + func_name;
		}
		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		this->shared_mem_ptr = data.comm_shared_mem;
		SPDLOG_INFO(
			"No explicit pass matched; recorded generic entry for {}",
			func_name);
		return id;
	}
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

	void *register_fatbin_end_addr =
		GSIZE_TO_POINTER(gum_module_find_export_by_name(
			nullptr, "__cudaRegisterFatBinaryEnd"));
	register_hook(AttachedToFunction::RegisterFatbinEnd,
		      register_fatbin_end_addr);

	void *cuda_launch_kernel_addr = GSIZE_TO_POINTER(
		gum_module_find_export_by_name(nullptr, "cudaLaunchKernel"));
	register_hook(AttachedToFunction::CudaLaunchKernel,
		      cuda_launch_kernel_addr);

	gum_interceptor_end_transaction(interceptor);

	// Discover pass definitions from directory
	try {
		const char *env_dir = std::getenv("BPFTIME_PTXPASS_DIR");
		std::string scan_dir = env_dir && std::strlen(env_dir) > 0 ?
					       std::string(env_dir) :
					       std::string(DEFAULT_PASSES_DIR);
		this->pass_definitions =
			load_pass_definitions_from_dir(scan_dir);
		SPDLOG_INFO("Discovered {} pass definitions from {}",
			    this->pass_definitions.size(), scan_dir.c_str());
	} catch (const std::exception &e) {
		SPDLOG_ERROR("Failed to load pass definitions: {}", e.what());
	}
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

	// New: run discovered JSON-based pass executables in priority order
	auto &to_patch_ptx = ptx_out[0];
	to_patch_ptx = filter_unprintable_chars(to_patch_ptx);
	const std::string map_sym = "map_info";
	const std::string const_sym = "constData";
	try {
		// Build ordered list of (priority, id) for entries with
		// configured pass
		struct Item {
			int priority;
			int id;
		};
		std::vector<Item> items;
		for (const auto &kv : hook_entries) {
			const auto &e = kv.second;
			if (!e.pass_exec.empty())
				items.push_back(Item{ e.priority, kv.first });
		}
		std::sort(items.begin(), items.end(),
			  [](const Item &a, const Item &b) {
				  if (a.priority != b.priority)
					  return a.priority < b.priority;
				  return a.id < b.id;
			  });
		for (const auto &it : items) {
			const auto &e = hook_entries.at(it.id);
			// Prefer attach_point_override when present
			std::string attach_point;
			if (e.attach_point_override.has_value())
				attach_point = *e.attach_point_override;
			// fallback: derive from parameters if needed (left as
			// future extension) For each kernel, run the pass once;
			// replace PTX if output is non-empty
			const auto &kernels =
				e.kernels.empty() ? std::vector<std::string>{} :
						    e.kernels;
			if (kernels.empty()) {
				// Fallback to function name if provided in
				// parameters
				auto itf = e.parameters.find("func_name");
				if (itf != e.parameters.end()) {
					auto res = run_pass_executable_json(
						e.pass_exec, to_patch_ptx,
						itf->second, map_sym,
						const_sym);
					if (res.has_value() && !res->empty())
						to_patch_ptx = *res;
				}
				continue;
			}
			for (const auto &k : kernels) {
				auto res = run_pass_executable_json(
					e.pass_exec, to_patch_ptx, k, map_sym,
					const_sym);
				if (!res.has_value()) {
					SPDLOG_WARN(
						"Pass {} failed for kernel {}",
						e.pass_exec, k);
					continue;
				}
				if (!res->empty())
					to_patch_ptx = *res;
			}
		}
	} catch (const std::exception &e) {
		SPDLOG_ERROR("Pass execution error: {}", e.what());
		return {};
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
	std::string command =
		"nvcc -O2 -G -g --keep-device-functions -arch=sm_60 ";
	{
		auto ptx_in = work_dir / "main.ptx";
		// SPDLOG_WARN("Using /tmp/main.ptx as ptx for nvcc");
		// std::string ptx_in = "/tmp/main.ptx";
		SPDLOG_INFO("PTX IN: {}", ptx_in.c_str());
		std::ofstream ofs(ptx_in);
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
// Ensure buffer size matches map_info layout used on device (256 entries)
static char map_basic_info_mock[sizeof(attach::MapBasicInfo) * 256];
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
			  (char *)"constData", "constData", 0,
			  sizeof(_constData_mock), 1, 0);

	__cudaRegisterVar(fatbin_handle, (char *)&map_basic_info_mock,
			  (char *)"map_info", "map_info", 0,
			  sizeof(map_basic_info_mock), 1, 0);
	this->trampoline_memory_state = TrampolineMemorySetupStage::Registered;
	SPDLOG_INFO("Register trampoline memory done");
	return 0;
}
int nv_attach_impl::copy_data_to_trampoline_memory()
{
	SPDLOG_INFO("Copying data to device symbols..");
	size_t const_size = 0;
	if (auto err = cudaGetSymbolSize(&const_size,
					 (const void *)&_constData_mock);
	    err != cudaSuccess || const_size < sizeof(_constData_mock)) {
		SPDLOG_ERROR(
			"cudaGetSymbolSize(constData) failed or size too small: {}",
			(int)err);
		return -1;
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
	if (!this->map_basic_info.has_value()) {
		SPDLOG_ERROR(
			"map_basic_info is not set, cannot copy to device");
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
	{
		size_t map_info_symbol_size = 0;
		if (auto err = cudaGetSymbolSize(
			    &map_info_symbol_size,
			    (const void *)&map_basic_info_mock);
		    err != cudaSuccess) {
			SPDLOG_ERROR("cudaGetSymbolSize(map_info) failed: {}",
				     (int)err);
			return -1;
		}
		size_t host_bytes = this->map_basic_info->size() *
				    sizeof(this->map_basic_info->at(0));
		size_t copy_bytes = host_bytes < map_info_symbol_size ?
					    host_bytes :
					    map_info_symbol_size;
		if (auto err = cudaMemcpyToSymbol(
			    (const void *)&map_basic_info_mock,
			    this->map_basic_info->data(), copy_bytes);
		    err != cudaSuccess) {
			SPDLOG_ERROR(
				"Unable to copy `map_basic_info` to device : {}",
				(int)err);
			return -1;
		}
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
		// In new flow, directly_run is not supported and should be
		// represented by a dedicated pass
		insts = itr->second.instuctions;
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
