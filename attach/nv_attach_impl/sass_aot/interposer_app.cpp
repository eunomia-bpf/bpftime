// PTX-free fixture application for launch-boundary interposition timing.
//
// A minimal first-party CUDA Driver API application: it loads its own
// SASS-only (PTX-free) cubin and repeatedly launches its own kernel. The
// application never calls any sass_aot function; when the
// bpftime_sass_aot_interposer library is LD_PRELOADed, the verified
// BPF-derived SASS callback is executed from within cuLaunchKernel
// instead, at the launch boundary.
//
// Usage: bpftime_sass_aot_interposer_app <app-cubin> [iterations]
//                                              [device-ordinal]
//
// Every numeric timing is preserved, one line per iteration:
//   iter <i> launch_ns <n> sync_ns <n> total_ns <n>
//   result <value>

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

[[noreturn]] void fail(const std::string &message)
{
	std::cerr << "interposer_app: " << message << '\n';
	std::exit(1);
}

std::string cuda_error(CUresult error, const char *operation)
{
	const char *name = nullptr;
	const char *description = nullptr;
	(void)cuGetErrorName(error, &name);
	(void)cuGetErrorString(error, &description);
	return std::string(operation) +
	       " failed: " + (name != nullptr ? name : "CUDA_ERROR_UNKNOWN") +
	       " (" +
	       (description != nullptr ? description : "no description") + ")";
}

int64_t now_ns()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		   std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

} // namespace

int main(int argc, char **argv)
{
	if (argc < 2 || argc > 4) {
		std::cerr << "usage: " << argv[0]
			  << " <app-cubin> [iterations] [device-ordinal]\n";
		return 2;
	}

	const std::string app_cubin = argv[1];
	int iterations = 4;
	int device_ordinal = 0;
	if (argc >= 3) {
		char *end = nullptr;
		const long value = std::strtol(argv[2], &end, 10);
		if (end == argv[2] || *end != '\0' || value <= 0) {
			std::cerr << "iterations must be a positive integer\n";
			return 2;
		}
		iterations = static_cast<int>(value);
	}
	if (argc >= 4) {
		char *end = nullptr;
		const long value = std::strtol(argv[3], &end, 10);
		if (end == argv[3] || *end != '\0' || value < 0) {
			std::cerr << "device ordinal must be non-negative\n";
			return 2;
		}
		device_ordinal = static_cast<int>(value);
	}

	CUcontext context = nullptr;
	CUmodule module = nullptr;
	CUfunction entry = nullptr;
	CUdeviceptr out = 0;
	bool allocated = false;

	if (const auto err = cuInit(0); err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuInit"));
	CUdevice device{};
	if (const auto err = cuDeviceGet(&device, device_ordinal);
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuDeviceGet"));
	if (const auto err = cuCtxCreate(&context, 0, device);
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuCtxCreate"));

	if (const auto err = cuModuleLoad(&module, app_cubin.c_str());
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuModuleLoad"));
	if (const auto err =
		    cuModuleGetFunction(&entry, module, "companion_app_kernel");
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuModuleGetFunction"));
	if (const auto err = cuMemAlloc(&out, sizeof(uint64_t));
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuMemAlloc"));
	allocated = true;

	void *args[] = { &out };
	for (int i = 0; i < iterations; ++i) {
		const int64_t t0 = now_ns();
		const CUresult launch =
			cuLaunchKernel(entry, 1, 1, 1, 1, 1, 1, 0, nullptr,
				       args, nullptr);
		const int64_t t1 = now_ns();
		if (launch != CUDA_SUCCESS) {
			if (allocated)
				(void)cuMemFree(out);
			(void)cuModuleUnload(module);
			(void)cuCtxDestroy(context);
			std::cerr << "interposer_app: "
				  << cuda_error(launch, "cuLaunchKernel")
				  << '\n';
			return 1;
		}
		const CUresult sync = cuCtxSynchronize();
		const int64_t t2 = now_ns();
		if (sync != CUDA_SUCCESS) {
			if (allocated)
				(void)cuMemFree(out);
			(void)cuModuleUnload(module);
			(void)cuCtxDestroy(context);
			std::cerr << "interposer_app: "
				  << cuda_error(sync, "cuCtxSynchronize") << '\n';
			return 1;
		}
		std::cout << "iter " << i << " launch_ns " << (t1 - t0)
			  << " sync_ns " << (t2 - t1) << " total_ns "
			  << (t2 - t0) << std::endl;
	}

	uint64_t result = 0;
	if (const auto err =
		    cuMemcpyDtoH(&result, out, sizeof(result));
	    err != CUDA_SUCCESS)
		fail(cuda_error(err, "cuMemcpyDtoH"));
	std::cout << "result " << result << std::endl;

	(void)cuMemFree(out);
	(void)cuModuleUnload(module);
	(void)cuCtxDestroy(context);
	return result == 7 ? 0 : 1;
}
