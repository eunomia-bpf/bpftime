#include "cuda_injector.hpp"
#include "memory_utils.hpp"
#include "spdlog/spdlog.h"
#include <sys/ptrace.h>
#include <regex>

using namespace bpftime;
using namespace attach;

// inline pos_retval_t __dispatch(pos_cli_options_t &clio)
// {
// 	switch (clio.action_type) {
// 	case kPOS_CliAction_Help:
// 		return handle_help(clio);

// 	case kPOS_CliAction_PreDump:
// 		return handle_predump(clio);

// 	case kPOS_CliAction_Dump:
// 		return handle_dump(clio);

// 	case kPOS_CliAction_Restore:
// 		return handle_restore(clio);

// 	case kPOS_CliAction_Migrate:
// 		return handle_migrate(clio);

// 	case kPOS_CliAction_TraceResource:
// 		return handle_trace(clio);

// 	case kPOS_CliAction_Start:
// 		return handle_start(clio);

// 	default:
// 		return POS_FAILED_NOT_IMPLEMENTED;
// 	}
// }

namespace oob_functions
{
POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_predump);
POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_dump);
POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_restore);
POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_trace_resource);
}; // namespace oob_functions

CUDAInjector::CUDAInjector(pid_t pid) : target_pid(pid)
{
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
		std::map<pos_oob_msg_typeid_t, oob_client_function_t>{
			{ kPOS_OOB_Msg_CLI_Ckpt_Dump,
			  oob_functions::cli_ckpt_dump::clnt } },
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
		std::map<pos_oob_msg_typeid_t, oob_client_function_t>{
			{ kPOS_OOB_Msg_CLI_Restore,
			  oob_functions::cli_restore::clnt } },
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

bool CUDAInjector::inject_ptx(const char *func_name, CUmodule &module)
{
	pos_retval_t retval = POS_SUCCESS;
	// retval = __dispatch(clio_checkpoint);
	// if (retval != POS_SUCCESS) {
	// 	return false;
	// }
	CUfunction target_addr;
	size_t dummy_code_size = sizeof(orig_ptx.c_str());
	CUmodule m;
	CUresult rc = cuModuleLoadData(&m, orig_ptx.c_str());
	rc = cuModuleGetFunction(&target_addr, m,
					func_name);

	// 2. Retrieve the function named "injected_kernel"
	CUfunction kernel;
	auto result = cuModuleGetFunction(&kernel, module, func_name);
	if (result != CUDA_SUCCESS) {
		spdlog::error("cuModuleGetFunction() failed: {}", (int)result);
		cuModuleUnload(module);
		return false;
	}

	// 3. Backup the original code
	CodeBackup backup;
	backup.addr = reinterpret_cast<CUdeviceptr>(target_addr);
	backups.push_back(backup);
	// how to push ptx into clio_restore
	// retval = __dispatch(clio_restore);
	// if (retval != POS_SUCCESS) {
	// 	return false;
	// }
	// need to hack the restored ptx code
	return true;
}
