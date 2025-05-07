
#include "driver_types.h"
#include "spdlog/spdlog.h"
#include "vector_types.h"
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <vector>
#include "ptx_load_mocker.hpp"
using namespace bpftime;
using namespace nvattach;

union load_mocker_container {
	load_mocker mocker;
	load_mocker *operator->()
	{
		return &mocker;
	}
	load_mocker_container()
	{
	}
	~load_mocker_container()
	{
	}
};
static load_mocker_container mocker;
static int ctx_initialized = 0;
static void initialize_ctx()
{
	int expected = 0;
	if (__atomic_compare_exchange_n(&ctx_initialized, &expected, 1, false,
					__ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
		new (&mocker.mocker) load_mocker;
	}
}
template <class T>
static inline T try_get_original_func(const char *name, T &store)
{
	if (store == nullptr) {
		store = (T)dlsym(RTLD_NEXT, name);
	}
	return store;
}


extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
				       const char *hostFun, char *deviceFun,
				       const char *deviceName, int thread_limit,
				       uint3 *tid, uint3 *bid, dim3 *bDim,
				       dim3 *gDim, int *wSize)
{
	initialize_ctx();
	auto orig = try_get_original_func("__cudaRegisterFunction",
					  mocker->orig___cudaRegisterFunction);
	SPDLOG_INFO(
		"Mock __cudaRegisterFunction: hostFun={:x}, deviceFun={}, deviceName={}, thread_limit={}",
		(uintptr_t)hostFun, deviceFun, deviceName, thread_limit);
	orig(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
	     bid, bDim, gDim, wSize);
	if (auto itr = mocker->fatbin_handle_to_data.find(fatCubinHandle);
	    itr != mocker->fatbin_handle_to_data.end()) {
		itr->second.host_funcs.push_back((void *)hostFun);
		mocker->host_func_to_device_func[(void *)hostFun] =
			host_wrapper_func{ .device_name =
						   std::string(deviceName),
					   .device_symbol =
						   std::string(deviceFun),
					   .fatbin_handle = fatCubinHandle };
		SPDLOG_INFO("Mock __cudaRegisterFunction: Recorded");
	} else {
		SPDLOG_INFO(
			"Mock __cudaRegisterFunction: Invalid handle, ignoring");
	}
}
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
	initialize_ctx();
	auto orig =
		try_get_original_func("__cudaRegisterFatBinaryEnd",
				      mocker->orig___cudaRegisterFatBinaryEnd);
	SPDLOG_INFO("Ending register for handle {:x}",
		    (uintptr_t)fatCubinHandle);
	orig(fatCubinHandle);
}
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
					dim3 blockDim, void **args,
					size_t sharedMem, cudaStream_t stream)
{
	initialize_ctx();
	auto orig = try_get_original_func("cudaLaunchKernel",
					  mocker->orig_cudaLaunchKernel);
	SPDLOG_INFO("Mock cudaLaunchKernel: host func {:x}", (uintptr_t)func);
	auto err = orig(func, gridDim, blockDim, args, sharedMem, stream);
	if (err != cudaSuccess) {
		SPDLOG_INFO("Mock cudaLaunchKernel: unsuccessful, return");
		return err;
	}
	if (auto itr = mocker->host_func_to_device_func.find((void *)func);
	    itr != mocker->host_func_to_device_func.end()) {
		SPDLOG_INFO("Mock cudaLaunchKernel: Calling device func {}",
			    itr->second.device_name);
		mocker->last_executed_host_func = (void *)func;
	} else {
		SPDLOG_INFO("Mock cudaLaunchKernel: unrecorded host func: {:x}",
			    (uintptr_t)func);
	}
	return err;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	initialize_ctx();
	auto orig =
		try_get_original_func("__cudaUnregisterFatBinary",
				      mocker->orig___cudaUnregisterFatBinary);
	SPDLOG_INFO("Mock __cudaUnregisterFatBinary: Destroying handle {:x}",
		    (uintptr_t)fatCubinHandle);
	orig(fatCubinHandle);
	if (auto itr = mocker->fatbin_handle_to_data.find(fatCubinHandle);
	    itr != mocker->fatbin_handle_to_data.end()) {
		for (auto item : itr->second.host_funcs) {
			mocker->host_func_to_device_func.erase(item);
		}
		mocker->fatbin_handle_to_data.erase(itr);
	} else {
		SPDLOG_INFO("Mock __cudaUnregisterFatBinary: Invalid handle");
	}
}
