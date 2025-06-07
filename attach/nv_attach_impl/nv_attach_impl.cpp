#include "nv_attach_impl.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "frida-gum.h"

#include "llvm_jit_context.hpp"
#include "nv_attach_private_data.hpp"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers
#include <boost/process/detail/child_decl.hpp>
#include <boost/process/env.hpp>
#include <boost/process/io.hpp>
#include <boost/process/pipe.hpp>
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
// #include <pos/cuda_impl/utils/fatbin.h>
#include <unistd.h>
#include <variant>
#include <vector>
#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <sstream>
using namespace bpftime;
using namespace attach;

extern GType cuda_runtime_function_hooker_get_type();

std::string get_device_sm_version() {
	int device_count;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess || device_count == 0) {
		SPDLOG_WARN("No CUDA devices found, using default sm_90");
		return "sm_90";
	}

	cudaDeviceProp device_prop;
	err = cudaGetDeviceProperties(&device_prop, 0);  // Get properties of first device
	if (err != cudaSuccess) {
		SPDLOG_WARN("Failed to get device properties, using default sm_90");
		return "sm_90";
	}

	std::stringstream ss;
	ss << "sm_" << device_prop.major << device_prop.minor;
	SPDLOG_INFO("Detected CUDA device SM version: {}", ss.str());
	return ss.str();
}

int nv_attach_impl::detach_by_id(int id)
{
	SPDLOG_INFO("Detaching is not supported by nv_attach_impl");
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	// if (this->hook_entries.size() >= 1) {
	// 	SPDLOG_ERROR("Only one nv attach could be used");
	// 	return -1;
	// }
	auto data = dynamic_cast<const nv_attach_private_data &>(private_data);
	if (attach_type == ATTACH_CUDA_PROBE) {
		if (std::get<std::string>(data.code_addr_or_func_name) ==
		    "__memcapture") {
			SPDLOG_INFO("Recording memcapture in nv_attach_impl");
			auto id = this->allocate_id();
			hook_entries[id] = nv_attach_entry{
				.type = nv_attach_cuda_memcapture{},
				.instuctions = data.instructions,
				.kernels = data.func_names
			};
			this->map_basic_info = data.map_basic_info;
			this->shared_mem_ptr = data.comm_shared_mem;
			return id;
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
				.kernels = data.func_names
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
			.kernels = data.func_names
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
	// Lambda: 获取当前线程 TID（Linux 特有）
	// auto get_tid = []() -> pid_t {
	// 	return static_cast<pid_t>(syscall(SYS_gettid));
	// };

	// // Lambda: 读取 /proc/<pid>/task/<tid>/children，返回子进程 PID 列表
	// auto list_children = [](pid_t pid, pid_t tid) -> std::vector<pid_t> {
	// 	std::vector<pid_t> children;
	// 	std::string path = "/proc/" + std::to_string(pid) + "/task/" +
	// 			   std::to_string(tid) + "/children";

	// 	std::ifstream ifs(path);
	// 	if (!ifs.is_open()) {
	// 		std::perror(("open " + path).c_str());
	// 		return children;
	// 	}

	// 	std::string line;
	// 	if (std::getline(ifs, line)) {
	// 		std::istringstream iss(line);
	// 		pid_t cpid;
	// 		while (iss >> cpid) {
	// 			children.push_back(cpid);
	// 		}
	// 	}
	// 	return children;
	// };

	// pid_t pid = getpid(); // 当前进程 PID
	// pid_t tid = get_tid(); // 当前线程 TID

	// auto kids = list_children(pid, tid);
	// this->injector = std::make_unique<CUDAInjector>(kids[0]);
	// this->injector = std::make_unique<CUDAInjector>(getpid());

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
		boost::process::ipstream err_stream;
		boost::process::environment env =
			boost::this_process::environment();
		env["LD_PRELOAD"] = "";

		boost::process::child child(
			std::string("cuobjdump --dump-ptx ") +
				output_path.string(),
			boost::process::std_out > stream,
			boost::process::std_err > err_stream,
			boost::process::env(env));
		std::string line;
		std::string output;
		std::string error_output;
		bool should_record = false;
		while (stream && std::getline(stream, line)) {
			if (should_record) {
				output += line + "\n";
			}
			if (line.starts_with("compressed"))
				should_record = true;
		}
		while (err_stream && std::getline(err_stream, line)) {
			error_output += line + "\n";
		}
		child.wait();
		if (child.exit_code() != 0) {
			SPDLOG_ERROR("cuobjdump failed with exit code {}: {}", 
				child.exit_code(), error_output);
			return {};
		}
		if (!error_output.empty()) {
			SPDLOG_WARN("cuobjdump warnings: {}", error_output);
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
	std::string command =
		"nvcc -O2 -G -g --keep-device-functions -arch=" + get_device_sm_version() + " ";
	{
		auto ptx_in = work_dir / "main.ptx";
		// SPDLOG_WARN("Using /tmp/main.ptx as ptx for nvcc");
		// std::string ptx_in = "/tmp/main.ptx";
		SPDLOG_INFO("PTX IN: {}", ptx_in.c_str());
		std::ofstream ofs(ptx_in);
		ofs << ptx_out[0];
		command += ptx_in;
		command += " ";
	}
	command += "-fatbin ";
	auto fatbin_out = work_dir / "out.fatbin";
	command += "-o ";
	command += fatbin_out;
	SPDLOG_INFO("Fatbin out {}", fatbin_out.c_str());
	SPDLOG_INFO("Starting nvcc: {}", command);
	boost::process::ipstream nvcc_out;
	boost::process::ipstream nvcc_err;
	boost::process::child nvcc_child(
		command,
		boost::process::std_out > nvcc_out,
		boost::process::std_err > nvcc_err
	);
	
	std::string nvcc_output;
	std::string nvcc_error;
	std::string line;
	
	// Read stdout
	while (nvcc_out && std::getline(nvcc_out, line)) {
		nvcc_output += line + "\n";
	}
	
	// Read stderr
	while (nvcc_err && std::getline(nvcc_err, line)) {
		nvcc_error += line + "\n";
	}
	
	nvcc_child.wait();
	if (nvcc_child.exit_code() != 0) {
		SPDLOG_ERROR("nvcc failed with exit code {}: {}", 
			nvcc_child.exit_code(), nvcc_error);
		return {};
	}
	if (!nvcc_error.empty()) {
		SPDLOG_WARN("nvcc warnings: {}", nvcc_error);
	}
	if (!nvcc_output.empty()) {
		SPDLOG_INFO("nvcc output: {}", nvcc_output);
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
	if (auto err = cudaMemcpyToSymbol((const void *)&_constData_mock,
					  &this->shared_mem_ptr,
					  sizeof(_constData_mock));
	    err != cudaSuccess) {
		SPDLOG_ERROR(
			"Unable to copy `constData` (shared memory address) to device: {}",
			(int)err);
		return -1;
	}
	// Prefill some data
	// SPDLOG_WARN(
	// 	"Prefilling key_size & value_size to 4 for all map_basic_info");
	// for (auto &item : *this->map_basic_info) {
	// 	item.key_size = 4;
	// 	item.value_size = 4;
	// }
	SPDLOG_INFO("Copying the followed map info:");
	for (int i = 0; i < this->map_basic_info->size(); i++) {
		const auto &cur = this->map_basic_info->at(i);
		if (cur.enabled) {
			SPDLOG_INFO(
				"Mapid {}, enabled = {}, key_size = {}, value_size = {}, max_ent={}",
				i, cur.enabled, cur.key_size, cur.value_size,
				cur.max_entries);
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
	for (auto c : input) {
		if (set.contains(c) || (uint8_t)c > 127)
			continue;
		result.push_back(c);
	}
	
	// Only try to remove characters if the string is not empty
	if (!result.empty()) {
		SPDLOG_INFO("try to remove characters if the string is not empty: {}", result);
		// Find the last occurrence of '}'
		size_t last_brace = result.find_last_of('}');
		if (last_brace != std::string::npos) {
			// If we found a '}', keep everything up to and including it
			result = result.substr(0, last_brace + 1);
		}
	}
	return result;
};

} // namespace bpftime::attach
