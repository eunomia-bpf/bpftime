// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t error)
{
	if (error != hipSuccess)
		throw std::runtime_error(hipGetErrorString(error));
}
extern "C" __global__ void bpftime_example_kernel(unsigned *out, unsigned n)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	out[i] = i * 7 + 3;
}
int main(int argc, char **argv)
{
	try {
		unsigned blocks = argc > 1 ? std::stoul(argv[1]) : 1;
		unsigned repeats = argc > 2 ? std::stoul(argv[2]) : 2;
		unsigned streams = argc > 3 ? std::stoul(argv[3]) : 1;
		if (argc > 4 || !blocks || blocks > 64 || !repeats ||
		    repeats > 20 || !streams || streams > 4) {
			std::fprintf(
				stderr,
				"Usage: %s [blocks:1-64 [repeats:1-20 [streams:1-4]]]\n",
				argv[0]);
			return 2;
		}
		hipDeviceProp_t props{};
		check(hipGetDeviceProperties(&props, 0));
		unsigned threads = props.warpSize;
		unsigned n = blocks * threads - 1;
		std::vector<hipStream_t> queues(streams);
		std::vector<unsigned *> device(streams);
		for (unsigned s = 0; s < streams; ++s) {
			check(hipStreamCreate(&queues[s]));
			check(hipMalloc(&device[s],
					(n + 1) * sizeof(unsigned)));
		}
		for (unsigned r = 0; r < repeats; ++r)
			for (unsigned s = 0; s < streams; ++s) {
				hipLaunchKernelGGL(bpftime_example_kernel,
						   dim3(blocks), dim3(threads),
						   0, queues[s], device[s], n);
				check(hipGetLastError());
			}
		for (unsigned s = 0; s < streams; ++s) {
			check(hipStreamSynchronize(queues[s]));
			std::vector<unsigned> result(n);
			check(hipMemcpy(result.data(), device[s],
					n * sizeof(unsigned),
					hipMemcpyDeviceToHost));
			for (unsigned i = 0; i < n; ++i)
				if (result[i] != i * 7 + 3)
					throw std::runtime_error(
						"Kernel result changed");
			check(hipFree(device[s]));
			check(hipStreamDestroy(queues[s]));
		}
		std::printf("VICTIM OK waves=%u\n", blocks * repeats * streams);
		return 0;
	} catch (const std::exception &e) {
		std::fprintf(stderr, "%s\n", e.what());
		return 1;
	}
}
