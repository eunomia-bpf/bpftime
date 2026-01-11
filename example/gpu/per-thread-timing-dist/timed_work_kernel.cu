#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <unistd.h>

// Simple CUDA error checking
#define CHECK_CUDA(call)                                                       \
	do {                                                                   \
		cudaError_t err__ = (call);                                    \
		if (err__ != cudaSuccess) {                                    \
			fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",       \
				cudaGetErrorString(err__), err__, __FILE__,    \
				__LINE__);                                     \
			std::exit(EXIT_FAILURE);                               \
		}                                                              \
	} while (0)

// Kernel: each thread runs a synthetic workload
extern "C" __global__ void timed_work_kernel(int base_iters)
{
	const unsigned int globalThreadId =
		blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread runs a slightly different number of iterations
	// to produce a non-trivial distribution.
	// Create 5 distinct clusters of workload to show up in the histogram
	int scale = (globalThreadId % 5) + 1;
	int my_iters = base_iters * scale;

	// "work" section
	volatile float acc = 0.0f;
	for (int i = 0; i < my_iters; ++i) {
		acc += 1.0f; // trivial arithmetic
	}

	// Prevent the compiler from optimizing out the loop completely
	if (acc == -1.0f) {
		printf("This will never be printed\n");
	}
}

int main()
{
	// Again, keep it small for printing
	const int BLOCKS = 4;
	const int THREADS_PER_BLOCK = 64;

	const int BASE_ITERS = 100000; // base workload per thread

	printf("=== Per-thread runtime distribution demo (bpftime) ===\n");
	printf("Grid: %d blocks, Block: %d threads (total %d threads)\n",
	       BLOCKS, THREADS_PER_BLOCK, BLOCKS * THREADS_PER_BLOCK);
	printf("Starting to run the kernel in a loop. Stop with Ctrl+C.\n");

	// Launch kernel
	dim3 grid(BLOCKS);
	dim3 block(THREADS_PER_BLOCK);

	while (true) {
		std::printf("running\n");
		timed_work_kernel<<<grid, block>>>(BASE_ITERS);
		CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
		sleep(1);
	}

	return 0;
}