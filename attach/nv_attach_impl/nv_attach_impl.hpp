#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include <base_attach_impl.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <regex>
#include <cuda.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <fstream>
#include <utility>
#include "pos/include/workspace.h"
#include "pos/cuda_impl/remoting/workspace.h"

namespace bpftime
{
namespace attach
{

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;

// You would replace this with your own memory reading utility.
namespace memory_utils
{
static inline ssize_t process_vm_readv(pid_t pid, const struct iovec *local_iov,
				       unsigned long liovcnt,
				       const struct iovec *remote_iov,
				       unsigned long riovcnt,
				       unsigned long flags)
{
	return syscall(SYS_process_vm_readv, pid, local_iov, liovcnt,
		       remote_iov, riovcnt, flags);
}
template <typename T>
bool read_memory(pid_t pid, const void *remote_addr, T *out_value)
{
	// 首先尝试使用 process_vm_readv
	struct iovec local_iov = { .iov_base = out_value,
				   .iov_len = sizeof(T) };

	struct iovec remote_iov = { .iov_base = const_cast<void *>(remote_addr),
				    .iov_len = sizeof(T) };

	ssize_t read = process_vm_readv(pid, &local_iov, 1, &remote_iov, 1, 0);
	if (read == sizeof(T)) {
		return true;
	}

	// 如果 process_vm_readv 失败，尝试使用 ptrace
	// 注意：这种方法需要进程被暂停（通过 PTRACE_ATTACH 或其他方式）

	// 对于不同大小的数据类型，我们可能需要多次读取
	const size_t word_size = sizeof(long);
	const size_t num_words = (sizeof(T) + word_size - 1) / word_size;

	uint8_t *buffer = reinterpret_cast<uint8_t *>(out_value);
	uintptr_t addr = reinterpret_cast<uintptr_t>(remote_addr);

	for (size_t i = 0; i < num_words; ++i) {
		errno = 0;
		long word = ptrace(PTRACE_PEEKDATA, pid, addr + (i * word_size),
				   nullptr);

		if (errno != 0) {
			return false;
		}

		// 计算这个字应该复制多少字节
		size_t bytes_to_copy =
			std::min(word_size, sizeof(T) - (i * word_size));

		// 复制数据到输出缓冲区
		std::memcpy(buffer + (i * word_size), &word, bytes_to_copy);
	}

	return true;
}
} // namespace memory_utils

// ----------------------------------------------------------------------------
// A simple wrapper class to handle attaching to a CUDA context in another
// process. In a real scenario, you might separate this into its own .hpp/.cpp
// files.
// ----------------------------------------------------------------------------
class CUDAInjector {
    public:
	pid_t target_pid;
	CUcontext cuda_ctx{ nullptr };

	// Storing a backup of code, for illustration.
	// You can remove or adapt this if you don’t actually need code
	// injection.
	struct CodeBackup {
		CUdeviceptr addr;
	};
	std::vector<CodeBackup> backups;
	std::string orig_ptx;
	POSWorkspace_CUDA *ws = new POSWorkspace_CUDA();
	POSClient *client = new POSClient_CUDA();

	explicit CUDAInjector(pid_t pid, std::string orig_ptx) : target_pid(pid)
	{
		// Lambda 表达式：用于读取文件内容到 std::string
		auto orig_ptx_func =
			[](const std::string &filename) -> std::string {
			std::ifstream file(filename); // 打开文件输入流
			if (!file.is_open()) {
				throw std::ios_base::failure(
					"Failed to open the file: " + filename);
			}

			// 使用 stringstream 将文件内容加载为字符串
			std::ostringstream contentStream;
			contentStream << file.rdbuf();
			file.close();
			return contentStream.str();
		};
		this->orig_ptx = orig_ptx_func(orig_ptx);
		client->init(false);
		spdlog::debug("CUDAInjector: constructor for PID {}",
			      target_pid);

		// 检查目标进程是否存在
		if (kill(target_pid, 0) != 0) {
			throw std::runtime_error(
				"Target process does not exist");
		}

		// 初始化 CUDA Driver API
		CUresult res = cuInit(0);
		if (res != CUDA_SUCCESS) {
			const char *error_str;
			cuGetErrorString(res, &error_str);
			throw std::runtime_error(
				std::string("CUDA initialization failed: ") +
				error_str);
		}

		// 检查是否有可用的 CUDA 设备
		int device_count = 0;
		res = cuDeviceGetCount(&device_count);
		if (res != CUDA_SUCCESS || device_count == 0) {
			throw std::runtime_error("No CUDA devices available");
		}

		spdlog::debug(
			"CUDA initialized successfully with {} devices available",
			device_count);
		ws = pos_create_workspace_cuda();
		pos_create_client_param_t param = { .job_name = "bpftime",
						    .pid = target_pid,
						    .id = 1,
						    .is_restoring = false };
		ws->__create_client(param, &client);
	}

	bool attach()
	{
		spdlog::info("Attaching via PTRACE to PID {}", target_pid);
		if (ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr) == -1) {
			spdlog::error("PTRACE_ATTACH failed: {}",
				      strerror(errno));
			return false;
		}
		// Wait for the process to stop
		if (waitpid(target_pid, nullptr, 0) == -1) {
			spdlog::error("waitpid failed: {}", strerror(errno));
			return false;
		}

		// Attempt to locate and set the CUDA context in the target
		// process
		if (!get_cuda_context()) {
			spdlog::error(
				"Failed to get CUDA context from process {}",
				target_pid);
			return false;
		}

		spdlog::info("Attach to PID {} successful", target_pid);
		return true;
	}

	bool detach()
	{
		spdlog::info("Detaching via PTRACE from PID {}", target_pid);
		if (ptrace(PTRACE_DETACH, target_pid, nullptr, nullptr) == -1) {
			spdlog::error("PTRACE_DETACH failed: {}",
				      strerror(errno));
			return false;
		}
		return true;
	}

    private:
	// ------------------------------------------------------------------------
	// Below is minimal logic to demonstrate how you MIGHT find a CUDA
	// context. In reality, hooking into a remote process’s memory for CUDA
	// contexts is significantly more complex (symbol lookup, driver calls,
	// etc.).
	// ------------------------------------------------------------------------
	bool get_cuda_context()
	{
		// 首先尝试获取目标进程的 CUDA 驱动符号
		std::ifstream mapsFile("/proc/" + std::to_string(target_pid) +
				       "/maps");
		if (!mapsFile.is_open()) {
			spdlog::error("Failed to open /proc/{}/maps",
				      target_pid);
			return false;
		}

		// 先初始化我们自己的 CUDA 上下文
		CUdevice current_device;
		CUresult res = cuDeviceGet(&current_device, 0);
		if (res != CUDA_SUCCESS) {
			spdlog::error("Failed to get CUDA device");
			return false;
		}

		CUcontext our_context;
		res = cuCtxCreate(&our_context, 0, current_device);
		if (res != CUDA_SUCCESS) {
			spdlog::error("Failed to create CUDA context");
			return false;
		}

		std::string line;
		std::vector<std::pair<uintptr_t, uintptr_t> > cuda_regions;

		while (std::getline(mapsFile, line)) {
			if (line.find("libcuda.so") != std::string::npos) {
				uintptr_t start, end;
				if (sscanf(line.c_str(), "%lx-%lx", &start,
					   &end) == 2) {
					cuda_regions.push_back({ start, end });
				}
			}
		}

		// 对每个找到的 CUDA 内存区域进行扫描
		for (const auto &region : cuda_regions) {
			spdlog::debug("Scanning CUDA region: {:x}-{:x}",
				      region.first, region.second);

			for (uintptr_t addr = region.first;
			     addr < region.second; addr += sizeof(void *)) {
				CUcontext possible_ctx;
				if (!memory_utils::read_memory(target_pid,
							       (void *)addr,
							       &possible_ctx)) {
					continue;
				}

				// 跳过明显无效的指针
				if (possible_ctx == nullptr ||
				    reinterpret_cast<uintptr_t>(possible_ctx) <
					    0x1000) {
					continue;
				}

				if (validate_cuda_context(possible_ctx)) {
					spdlog::info(
						"Found valid CUDA context at remote address {:x}",
						addr);
					cuda_ctx = possible_ctx;
					return true;
				}
			}
		}

		spdlog::error("No valid CUDA context found in target process");
		return false;
	}

	bool validate_cuda_context(CUcontext remote_ctx)
	{
		// 不要直接使用远程进程的上下文
		CUcontext current_ctx = nullptr;
		CUresult res = cuCtxGetCurrent(&current_ctx);
		if (res != CUDA_SUCCESS) {
			spdlog::debug("No current CUDA context in our process");
			return false;
		}

		// 检查远程上下文是否是有效的指针
		if (remote_ctx == nullptr) {
			return false;
		}

		// 尝试读取远程上下文的一些基本信息
		CUdevice device;
		if (!memory_utils::read_memory(
			    target_pid, reinterpret_cast<void *>(remote_ctx),
			    &device)) {
			return false;
		}

		// 可以添加更多的验证逻辑
		int compute_capability_major = 0;
		res = cuDeviceGetAttribute(
			&compute_capability_major,
			CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		if (res != CUDA_SUCCESS) {
			return false;
		}

		spdlog::debug(
			"Found potential CUDA context with compute capability {}.x",
			compute_capability_major);
		return true;
	}

    public:
	// Demonstrates how you might inject PTX or backup/restore code on the
	// fly in a remote context. This is a stub for illustration.
	bool inject_ptx(const char *ptx_code1, CUdeviceptr target_addr,
			size_t code_size)
	{
		client->persist_handles(true);
		client->persist("/tmp/bpftime");

		// 1. Load the PTX into a module
		CUmodule module;
		std::string probe_func = ptx_code1;
		std::string modified_ptx = this->orig_ptx;
		std::string function_name_part ;
		// 使用正则表达式匹配probe函数
		std::regex probe_regex(
			R"(.entry probe_(.+)__cuda\(.*\) \{((.*\n)*)((.*ret;\s*\n)*)\})");
		std::smatch probe_match;

		if (std::regex_search(probe_func, probe_match, probe_regex)) {
			// 从匹配结果中提取函数名和函数体
			std::string function_body = probe_match[2];
			 function_name_part =
				probe_match[1]; // 获取函数名部分
			std::cout << "函数体: " << function_body
				  << "\n函数名部分: " << function_name_part
				  << std::endl;

			// 从函数体中移除ret指令
			std::string body_without_ret = function_body;
			size_t ret_pos = body_without_ret.find("ret;");
			if (ret_pos != std::string::npos) {
				// 找到ret;所在行的开始位置
				size_t line_start =
					body_without_ret.rfind('\n', ret_pos);
				if (line_start == std::string::npos)
					line_start = 0;
				else
					line_start++; // 跳过换行符

				// 找到ret;所在行的结束位置
				size_t line_end =
					body_without_ret.find('\n', ret_pos);
				if (line_end == std::string::npos)
					line_end = body_without_ret.length();
				else
					line_end++; // 包含换行符

				// 移除整行
				body_without_ret.erase(line_start,
						       line_end - line_start);
			}

			// 找到最后一个.reg声明的位置
			std::regex reg_pattern(R"(\.reg \.b64 \t%rd<\d+>;)");
			std::smatch reg_match;

			if (std::regex_search(this->orig_ptx, reg_match,
					      reg_pattern)) {
				// 找到匹配的位置
				size_t insert_pos = reg_match.position() +
						    reg_match.length();

				// 在.reg声明后插入probe函数体
				modified_ptx.insert(insert_pos,
						    "\n" + body_without_ret);
			}

			// 输出修改后的代码
			std::cout << "修改后的代码：\n"
				  << modified_ptx << std::endl;
		} else {
			std::cout << "正则表达式未能匹配probe函数" << std::endl;
		}
		// 使用正则表达式匹配retprobe函数
		std::regex probe_regex1(
			R"(.entry (retprobe)_(.+)__cuda\(.*\) \{((.*\n)*)((.*ret;\s*\n)*)\})");
		std::smatch probe_match1;

		if (std::regex_search(probe_func, probe_match1, probe_regex1)) {
			// 从匹配结果中提取函数名和函数体
			std::string probe_name = probe_match1[1]; // retprobe
			std::string function_name =
				probe_match1[2]; // infinite_kernel
			std::string function_body = probe_match1[3]; // 函数体

			std::cout << "探针名: " << probe_name << "\n"
				  << "函数名: " << function_name << "\n"
				  << "函数体: " << function_body << std::endl;

			// 从函数体中移除ret指令
			std::string body_without_ret = function_body;
			size_t ret_pos = body_without_ret.find("ret;");
			if (ret_pos != std::string::npos) {
				// 找到ret;所在行的开始位置
				size_t line_start =
					body_without_ret.rfind('\n', ret_pos);
				if (line_start == std::string::npos)
					line_start = 0;
				else
					line_start++; // 跳过换行符

				// 找到ret;所在行的结束位置
				size_t line_end =
					body_without_ret.find('\n', ret_pos);
				if (line_end == std::string::npos)
					line_end = body_without_ret.length();
				else
					line_end++; // 包含换行符

				// 移除整行
				body_without_ret.erase(line_start,
						       line_end - line_start);
			}

			// 找到目标函数的右花括号位置
			std::regex target_func_regex(R"(\.visible \.entry )" +
						     function_name +
						     R"(\(\)[\s\S]*?\})");
			std::smatch target_match;

			if (std::regex_search(modified_ptx, target_match,
					      target_func_regex)) {
				std::string target_func = target_match[0];
				size_t closing_brace_pos =
					target_match.position() +
					target_func.rfind("}");

				// 在右花括号前插入retprobe函数体
				modified_ptx.insert(closing_brace_pos,
						    "\n" + body_without_ret);

				// 输出修改后的代码
				std::cout << "修改后的代码：\n"
					  << modified_ptx << std::endl;
			} else {
				std::cout << "未能找到目标函数 "
					  << function_name << std::endl;
			}
		} else {
			std::cout << "正则表达式未能匹配retprobe函数"
				  << std::endl;
		}
		CUresult result =
			cuModuleLoadData(&module, modified_ptx.c_str());
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleLoadData() failed: {}",
				      (int)result);
			return false;
		}

		// 2. Retrieve the function named "injected_kernel"
		CUfunction kernel;
		result =
			cuModuleGetFunction(&kernel, module, "infinite_kernel");
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleGetFunction() failed: {}",
				      (int)result);
			cuModuleUnload(module);
			return false;
		}

		// 3. Backup the original code
		CodeBackup backup;
		backup.addr = target_addr;
		backups.push_back(backup);

		// 4. Retrieve the actual kernel code from the module’s global
		// space
		CUfunction func_addr;
		size_t func_size;
		result = cuModuleGetFunction(&func_addr, module, function_name_part.c_str());
		if (result != CUDA_SUCCESS) {
			spdlog::error("cuModuleGetFunction() failed: {}",
				      (int)result);
			cuModuleUnload(module);
			return false;
		}

		// Clean up
		// cuModuleUnload(module);
		client->init(true);
		client->restore_apicxts("/tmp/bpftime");
		client->restore_handles("/tmp/bpftime");
		// TODO redirect to new func_addr.
		return true;
	}
};

extern std::optional<class nv_attach_impl *> global_nv_attach_impl;
struct nv_hooker_func_t {
	void *func;
};

struct nv_attach_private_data final : public attach_private_data {
	// Saved module name
	pid_t pid;
	// function name to probe
	std::string filename;
	// initialize_from_string
	int initialize_from_string(const std::string_view &sv) override;
	std::string to_string() const override
	{
		return std::format("{}{}", filename, pid);
	};
};

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	// Dispatch a syscall from text transformer
	int64_t dispatch_nv(int64_t arg1, int64_t arg2, int64_t arg3,
			    int64_t arg4, int64_t arg5, int64_t arg6);
	// Set the function of calling original nv
	void set_original_nv_function(nv_hooker_func_t func)
	{
		orig_nv = func;
	}
	// Set this nv trace attach impl instance to the global ones, which
	// could be accessed by text segment transformer
	void set_to_global()
	{
		global_nv_attach_impl = this;
	}
	int detach_by_id(int id);
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl()
	{
	}
	// Forward declare the nested Impl struct
	struct Impl;

    private:
	// The original syscall function
	nv_hooker_func_t orig_nv = { nullptr };
};

} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
