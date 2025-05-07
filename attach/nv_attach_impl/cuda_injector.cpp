#include "cuda_injector.hpp"
#include "memory_utils.hpp"
#include "spdlog/spdlog.h"
#include <sys/ptrace.h>
#include <regex>

using namespace bpftime;
using namespace attach;

inline pos_retval_t __dispatch(pos_cli_options_t &clio){
    switch (clio.action_type)
    {
    case kPOS_CliAction_Help:
        return handle_help(clio);

    case kPOS_CliAction_PreDump:
        return handle_predump(clio);

    case kPOS_CliAction_Dump:
        return handle_dump(clio);

    case kPOS_CliAction_Restore:
        return handle_restore(clio);

    case kPOS_CliAction_Migrate:
        return handle_migrate(clio);

    case kPOS_CliAction_TraceResource:
        return handle_trace(clio);

    case kPOS_CliAction_Start:
        return handle_start(clio);

    default:
        return POS_FAILED_NOT_IMPLEMENTED;
    }
}

namespace oob_functions {
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_predump);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_dump);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_restore);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_trace_resource);
}; // namespace oob_functions

CUDAInjector::CUDAInjector(pid_t pid, std::string orig_ptx) : target_pid(pid)
{
	// Lambda 表达式：用于读取文件内容到 std::string
	auto orig_ptx_func = [](const std::string &filename) -> std::string {
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
	SPDLOG_DEBUG("CUDAInjector: constructor for PID {}", target_pid);

	// 检查目标进程是否存在
	if (kill(target_pid, 0) != 0) {
		throw std::runtime_error("Target process does not exist");
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

	spdlog::debug("CUDA initialized successfully with {} devices available",
		      device_count);

	clio_checkpoint.action_type = kPOS_CliAction_Dump;
	clio_checkpoint.record_raw(static_cast<pos_cli_meta>(kPOS_CliMeta_Dir),
				   "/tmp/bpftime");
	clio_checkpoint.record_raw(static_cast<pos_cli_meta>(kPOS_CliMeta_Pid),
				   std::to_string(target_pid));
	clio_checkpoint.local_oob_client = new POSOobClient(
		/* req_functions */
		std::map<pos_oob_msg_typeid_t, oob_client_function_t> {
			{ kPOS_OOB_Msg_CLI_Ckpt_Dump,
			  oob_functions::cli_ckpt_dump::clnt }
		},
		/* local_port */ 10087,
		/* local_ip */ "0.0.0.0");
	POS_CHECK_POINTER(clio_checkpoint.local_oob_client);

	clio_restore.action_type = kPOS_CliAction_Restore;
	clio_restore.record_raw(static_cast<pos_cli_meta>(kPOS_CliMeta_Dir),
				"/tmp/bpftime");
	clio_restore.record_raw(static_cast<pos_cli_meta>(kPOS_CliMeta_Pid),
				std::to_string(target_pid));
	clio_restore.local_oob_client = new POSOobClient(
		/* req_functions */
		std::map<pos_oob_msg_typeid_t, oob_client_function_t> {
			{ kPOS_OOB_Msg_CLI_Restore,
			  oob_functions::cli_restore::clnt }
		},
		/* local_port */ 10088,
		/* local_ip */ "0.0.0.0");
	POS_CHECK_POINTER(clio_restore.local_oob_client);
}

bool CUDAInjector::attach()
{
	spdlog::info("Attaching via PTRACE to PID {}", target_pid);
	if (ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr) == -1) {
		spdlog::error("PTRACE_ATTACH failed: {}", strerror(errno));
		return false;
	}
	// Wait for the process to stop
	if (waitpid(target_pid, nullptr, 0) == -1) {
		spdlog::error("waitpid failed: {}", strerror(errno));
		return false;
	}

	spdlog::info("Attach to PID {} successful", target_pid);
	return true;
}

bool CUDAInjector::detach()
{
	spdlog::info("Detaching via PTRACE from PID {}", target_pid);
	if (ptrace(PTRACE_DETACH, target_pid, nullptr, nullptr) == -1) {
		spdlog::error("PTRACE_DETACH failed: {}", strerror(errno));
		return false;
	}
	return true;
}

bool CUDAInjector::validate_cuda_context(CUcontext remote_ctx)
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
	if (!memory_utils::read_memory(target_pid,
				       reinterpret_cast<void *>(remote_ctx),
				       &device)) {
		return false;
	}

	// 可以添加更多的验证逻辑
	int compute_capability_major = 0;
	res = cuDeviceGetAttribute(&compute_capability_major,
				   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				   device);
	if (res != CUDA_SUCCESS) {
		return false;
	}

	spdlog::debug(
		"Found potential CUDA context with compute capability {}.x",
		compute_capability_major);
	return true;
}

bool CUDAInjector::inject_ptx(const char *ptx_code1, CUdeviceptr target_addr,
			      size_t code_size, CUmodule &module)
{
	pos_retval_t retval = POS_SUCCESS;
	retval = __dispatch(clio_checkpoint);

	// 1. Load the PTX into a module
	std::string probe_func = ptx_code1;
	std::string modified_ptx = this->orig_ptx;
	std::string function_name_part;
	// 使用正则表达式匹配probe函数
	std::regex probe_regex(
		R"(.entry probe_(.+)__cuda\(.*\) \{((.*\n)*)((.*ret;\s*\n)*)\})");
	std::smatch probe_match;

	if (std::regex_search(probe_func, probe_match, probe_regex)) {
		// 从匹配结果中提取函数名和函数体
		std::string function_body = probe_match[2];
		function_name_part = probe_match[1]; // 获取函数名部分
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
			size_t line_end = body_without_ret.find('\n', ret_pos);
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

		if (std::regex_search(this->orig_ptx, reg_match, reg_pattern)) {
			// 找到匹配的位置
			size_t insert_pos =
				reg_match.position() + reg_match.length();

			// 在.reg声明后插入probe函数体
			modified_ptx.insert(insert_pos,
					    "\n" + body_without_ret);
		}

		// 输出修改后的代码
		std::cout << "修改后的代码：\n" << modified_ptx << std::endl;
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
		std::string function_name = probe_match1[2]; // infinite_kernel
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
			size_t line_end = body_without_ret.find('\n', ret_pos);
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
			size_t closing_brace_pos = target_match.position() +
						   target_func.rfind("}");

			// 在右花括号前插入retprobe函数体
			modified_ptx.insert(closing_brace_pos,
					    "\n" + body_without_ret);

			// 输出修改后的代码
			std::cout << "修改后的代码：\n"
				  << modified_ptx << std::endl;
		} else {
			std::cout << "未能找到目标函数 " << function_name
				  << std::endl;
		}
	} else {
		std::cout << "正则表达式未能匹配retprobe函数" << std::endl;
	}
	CUresult result = cuModuleLoadData(&module, modified_ptx.c_str());
	if (result != CUDA_SUCCESS) {
		spdlog::error("cuModuleLoadData() failed: {}", (int)result);
		return false;
	}

	// 2. Retrieve the function named "injected_kernel"
	CUfunction kernel;
	result = cuModuleGetFunction(&kernel, module, "infinite_kernel");
	if (result != CUDA_SUCCESS) {
		spdlog::error("cuModuleGetFunction() failed: {}", (int)result);
		cuModuleUnload(module);
		return false;
	}

	// 3. Backup the original code
	CodeBackup backup;
	backup.addr = target_addr;
	backups.push_back(backup);

	retval = __dispatch(clio_restore);
	// need to hack the restored ptx code
	return true;
}
