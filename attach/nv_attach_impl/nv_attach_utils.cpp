#include "nv_attach_utils.hpp"
#include "trampoline_ptx.h"
#include <cstring>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <openssl/sha.h>
#include <sstream>
#include <cstdlib>
#include <cuda.h>
namespace bpftime
{
namespace attach
{
std::string get_default_trampoline_ptx()
{
	return TRAMPOLINE_PTX;
}
std::string wrap_ptx_with_trampoline(std::string input)
{
	return get_default_trampoline_ptx() + input;
}

std::string rewrite_ptx_target(std::string ptx,
			       const std::string &sm_arch)
{
	if (sm_arch.empty())
		return ptx;

	// Rewrite .version based on SM arch (sm_120+ needs 8.7, sm_100+ needs 8.5)
	int sm_num = (sm_arch.size() > 3) ? std::atoi(sm_arch.c_str() + 3) : 0;
	const char *ptx_ver = (sm_num >= 120) ? "8.7" : (sm_num >= 100) ? "8.5" : nullptr;
	if (ptx_ver) {
		auto vpos = ptx.find(".version");
		if (vpos != std::string::npos) {
			vpos += 8;
			while (vpos < ptx.size() && ptx[vpos] == ' ') vpos++;
			auto vstart = vpos;
			while (vpos < ptx.size() && ptx[vpos] != '\n' && ptx[vpos] != ' ') vpos++;
			if (vpos > vstart)
				ptx.replace(vstart, vpos - vstart, ptx_ver);
		}
	}

	// Rewrite .target
	auto pos = ptx.find(".target");
	if (pos == std::string::npos)
		return ptx;
	pos += strlen(".target");
	while (pos < ptx.size() && (ptx[pos] == ' ' || ptx[pos] == '\t'))
		pos++;
	auto start = pos;
	while (pos < ptx.size() && ptx[pos] != ' ' && ptx[pos] != '\t' &&
	       ptx[pos] != '\n' && ptx[pos] != '\r' && ptx[pos] != ',')
		pos++;
	if (pos <= start)
		return ptx;
	ptx.replace(start, pos - start, sm_arch);
	return ptx;
}

std::string wrap_ptx_with_trampoline_for_sm(std::string input,
					    const std::string &sm_arch)
{
	return rewrite_ptx_target(get_default_trampoline_ptx(), sm_arch) + input;
}

std::string patch_main_from_func_to_entry(std::string result)
{
	const std::string entry_func = ".visible .func bpf_main";

	auto idx = result.find(entry_func);
	SPDLOG_INFO("entry_func ({}) index {}", entry_func, idx);

	if (idx != result.npos) {
		result = result.replace(idx, entry_func.size(),
					".visible .entry bpf_main");
	}
	return result;
}
std::string sha256(const void *data, size_t length)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	SHA256((unsigned char *)data, length, hash);

	std::stringstream ss;
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		ss << std::hex << std::setw(2) << std::setfill('0')
		   << (int)hash[i];
	}
	return ss.str();
}

std::string get_gpu_sm_arch()
{
	// First check environment variable
	const char *sm_arch_env = std::getenv("BPFTIME_SM_ARCH");
	if (sm_arch_env && sm_arch_env[0] != '\0') {
		SPDLOG_INFO("Using SM arch from BPFTIME_SM_ARCH: {}",
			    sm_arch_env);
		return sm_arch_env;
	}

	// Auto-detect from CUDA driver
	CUdevice device;
	int major = 0, minor = 0;

	// Prefer current context if one exists; otherwise fall back to device 0.
	CUresult err = cuCtxGetDevice(&device);
	if (err != CUDA_SUCCESS) {
		SPDLOG_INFO(
			"cuCtxGetDevice failed with {}, falling back to device 0 for SM arch detection",
			(int)err);

		err = cuInit(0);
		if (err != CUDA_SUCCESS) {
			SPDLOG_WARN(
				"Failed to initialize CUDA driver ({}), falling back to sm_61",
				(int)err);
			return "sm_61";
		}

		err = cuDeviceGet(&device, 0);
		if (err != CUDA_SUCCESS) {
			SPDLOG_WARN(
				"Failed to get CUDA device 0 ({}), falling back to sm_61",
				(int)err);
			return "sm_61";
		}
	}

	err = cuDeviceGetAttribute(&major,
				   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				   device);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Failed to get compute capability major ({}), falling back to sm_61",
			(int)err);
		return "sm_61";
	}

	err = cuDeviceGetAttribute(&minor,
				   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				   device);
	if (err != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"Failed to get compute capability minor ({}), falling back to sm_61",
			(int)err);
		return "sm_61";
	}

	std::string sm_arch = "sm_" + std::to_string(major * 10 + minor);
	SPDLOG_INFO("Auto-detected GPU SM arch: {} (compute capability {}.{})",
		    sm_arch, major, minor);

	if (setenv("BPFTIME_SM_ARCH", sm_arch.c_str(), 1) != 0) {
		SPDLOG_WARN(
			"Failed to set BPFTIME_SM_ARCH environment variable to {}",
			sm_arch);
	}

	return sm_arch;
}

} // namespace attach
} // namespace bpftime
