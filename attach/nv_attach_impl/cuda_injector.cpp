#include "cuda_injector.hpp"
#include "memory_utils.hpp"
#include "spdlog/spdlog.h"
#include <sys/ptrace.h>
#include <regex>

using namespace bpftime;
using namespace attach;

inline pos_retval_t __dispatch(pos_cli_options_t &clio)
{
	switch (clio.action_type) {
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
bool CUDAInjector::inject_ptx()
{
	pos_retval_t retval = POS_SUCCESS;
	retval = __dispatch(clio_checkpoint);
	if (retval != POS_SUCCESS) {
		return false;
	}

	retval = __dispatch(clio_restore);
	if (retval != POS_SUCCESS) {
		return false;
	}
	return true;
}
