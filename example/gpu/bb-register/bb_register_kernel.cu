#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <unistd.h>

__device__ __noinline__ unsigned long long bb_reg_odd_path(unsigned int marker)
{
	// Keep odd path semantically distinct to prevent if-conversion.
	return ((unsigned long long)marker) ^ 0x1234ull;
}

extern "C" __global__ void bb_reg_kernel(unsigned long long *out)
{
	unsigned int marker = 0xdeadbeefu;
	marker ^= (unsigned int)threadIdx.x;
	// Force a real control-flow split so BB1/BB2 exist for tracepoint attach.
	if ((threadIdx.x & 1u) != 0u) {
		out[threadIdx.x] = bb_reg_odd_path(marker);
		return;
	}

	marker += 1u;
	out[threadIdx.x] = (unsigned long long)marker;
}

int main()
{
	constexpr int kThreads = 1;
	unsigned long long *d_out = nullptr;
	unsigned long long h_out[kThreads] = {};

	cudaError_t err = cudaMalloc(&d_out, sizeof(h_out));
	if (err != cudaSuccess) {
		std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	for (int iter = 0;; iter++) {
		bb_reg_kernel<<<1, kThreads>>>(d_out);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
				     cudaGetErrorString(err));
			break;
		}

		err = cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
			break;
		}

		std::printf("iter=%d lane0=0x%llx expected_lane0=0xdeadbef0\n", iter,
			    h_out[0]);
		std::fflush(stdout);
		sleep(1);
	}

	cudaFree(d_out);
	return 0;
}
