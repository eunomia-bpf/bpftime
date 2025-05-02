#ifndef _PTX_LOAD_MOCKER_HPP
#define _PTX_LOAD_MOCKER_HPP

#include "driver_types.h"
#include "vector_types.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace bpftime
{
namespace nvattach
{
extern "C" {
typedef struct {
	int magic;
	int version;
	const unsigned long long *data;
	void *filename_or_fatbins;

} __fatBinC_Wrapper_t;
}

struct fatbin_data {
	std::vector<char> binary;
	std::vector<void *> host_funcs;
};

struct host_wrapper_func {
	std::string device_name;
	std::string device_symbol;
	void **fatbin_handle;
};

struct load_mocker {
	std::map<void **, fatbin_data> fatbin_handle_to_data;

	std::map<void *, host_wrapper_func> host_func_to_device_func;


	void **(*orig___cudaRegisterFatBinary)(void *) = nullptr;

	void (*orig___cudaRegisterFunction)(
		void **fatCubinHandle, const char *hostFun, char *deviceFun,
		const char *deviceName, int thread_limit, uint3 *tid,
		uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
	void (*orig___cudaRegisterFatBinaryEnd)(void **fatCubinHandle);

	cudaError_t (*orig_cudaLaunchKernel)(const void *func, dim3 gridDim,
					     dim3 blockDim, void **args,
					     size_t sharedMem,
					     cudaStream_t stream);

	void (*orig___cudaUnregisterFatBinary)(void **fatCubinHandle);

	void *last_executed_host_func = nullptr;
	std::optional<host_wrapper_func> get_last_executed_func()
	{
		if (!last_executed_host_func)
			return {};
		return host_func_to_device_func.at(last_executed_host_func);
	}
};

} // namespace nvattach
} // namespace bpftime

#endif
