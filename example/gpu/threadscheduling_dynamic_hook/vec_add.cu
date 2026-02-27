#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>
// A simple vector addition kernel
// This kernel is instrumented by bpftime to collect sm/warp/lane mapping
__global__
void vectorAdd(const float *A, const float *B, float *C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		C[idx] = A[idx] + B[idx];
	}
}

int main(int argc, char *argv[])
{
	// Configuration - can be adjusted to test different thread distributions
	int num_blocks = 4;      // Number of blocks
	int threads_per_block = 64;  // Threads per block (should be multiple of 32 for warp alignment)

	// Parse command line arguments
	if (argc >= 2) {
		num_blocks = atoi(argv[1]);
	}
	if (argc >= 3) {
		threads_per_block = atoi(argv[2]);
	}

	printf("SM/Warp/Lane Mapping Test\n");
	printf("=========================\n");
	printf("Blocks: %d, Threads per block: %d\n", num_blocks, threads_per_block);
	printf("Total threads: %d\n", num_blocks * threads_per_block);
	printf("Warps per block: %d\n", (threads_per_block + 31) / 32);
	printf("\n");

	// Get device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("GPU: %s\n", prop.name);
	printf("SMs: %d\n", prop.multiProcessorCount);
	printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Warp size: %d\n", prop.warpSize);
	printf("\n");

	// Set vector size
	const int h_N = num_blocks * threads_per_block;

	size_t bytes = h_N * sizeof(float);

	// Allocate and initialize host memory
	std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);

	for (int i = 0; i < h_N; ++i) {
		h_A[i] = float(i);
		h_B[i] = float(2 * i);
	}

	// Allocate device memory
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	// Copy to device
	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	printf("Starting kernel loop (Ctrl+C to stop)...\n\n");

	// Run the kernel in a loop
	int iteration = 0;
	while (true) {
		// Zero output array
		cudaMemset(d_C, 0, bytes);

		// Launch kernel
		vectorAdd<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, h_N);
		cudaDeviceSynchronize();

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
			break;
		}

		// Copy result back to host
		cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

		// Print status every iteration
		iteration++;
		printf("\rIteration %d: C[0]=%.0f, C[1]=%.0f (expected 0, 3)",
		       iteration, h_C[0], h_C[1]);
		fflush(stdout);

		// Sleep for 2 seconds between iterations
		sleep(2);
	}

	// Cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
