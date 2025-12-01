/*
 * vec_add.cu - Simple CUDA vector addition for testing GPU Host Maps
 *
 * This program repeatedly executes a vector addition kernel,
 * triggering the BPF probes that update the Host-backed maps.
 */
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

__constant__ int d_N;

// A simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_N) {
		C[idx] = A[idx] + B[idx];
	}
}

int main(int argc, char **argv)
{
	// Parse thread count from command line
	int threads_per_block = 10;  // Default
	int num_blocks = 1;
	int sleep_ms = 1000;  // Default sleep time between iterations

	if (argc > 1) {
		threads_per_block = atoi(argv[1]);
	}
	if (argc > 2) {
		num_blocks = atoi(argv[2]);
	}
	if (argc > 3) {
		sleep_ms = atoi(argv[3]);
	}

	int total_threads = threads_per_block * num_blocks;
	printf("CUDA Vector Addition - Host Map Test\n");
	printf("=====================================\n");
	printf("Threads per block: %d\n", threads_per_block);
	printf("Number of blocks: %d\n", num_blocks);
	printf("Total threads: %d\n", total_threads);
	printf("Sleep between iterations: %d ms\n", sleep_ms);
	printf("\n");

	// Set vector size in constant memory
	const int h_N = 1 << 20; // 1M elements
	cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));

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

	// Run the kernel in an infinite loop
	int iteration = 0;
	while (true) {
		iteration++;

		// Zero output array
		cudaMemset(d_C, 0, bytes);

		// Launch kernel with specified configuration
		vectorAdd<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();

		// Copy result back to host
		cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

		// Print status
		printf("[Iteration %d] Kernel completed with %d threads\n",
		       iteration, total_threads);
		printf("  C[0] = %.0f (expected 0)\n", h_C[0]);
		printf("  C[1] = %.0f (expected 3)\n", h_C[1]);

		// Sleep
		usleep(sleep_ms * 1000);
	}

	// Cleanup (unreachable in infinite loop)
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
