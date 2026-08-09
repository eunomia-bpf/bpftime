#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include "benchmark_common.h"

__constant__ int d_N;

// A simple vector addition kernel for benchmarking
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_N) {
		C[idx] = A[idx] + B[idx];
	}
}

int main(int argc, char **argv)
{
	// Parse command line arguments
	BenchmarkArgs args = BenchmarkArgs::parse(argc, argv);
	int h_N = args.size;
	int iterations = args.iterations;

	// Set vector size in constant memory
	cudaError_t err = cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpyToSymbol failed: "
			  << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	size_t bytes = h_N * sizeof(float);

	// Allocate and initialize host memory
	auto host_alloc_start = std::chrono::high_resolution_clock::now();
	std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);

	for (int i = 0; i < h_N; ++i) {
		h_A[i] = float(i);
		h_B[i] = float(2 * i);
	}
	auto host_alloc_end = std::chrono::high_resolution_clock::now();
	auto host_alloc_time =
		std::chrono::duration_cast<std::chrono::microseconds>(
			host_alloc_end - host_alloc_start)
			.count();

	// Allocate Device memory
	auto device_alloc_start = std::chrono::high_resolution_clock::now();
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);
	auto device_alloc_end = std::chrono::high_resolution_clock::now();
	auto device_alloc_time =
		std::chrono::duration_cast<std::chrono::microseconds>(
			device_alloc_end - device_alloc_start)
			.count();

	// Copy to device
	auto h2d_start = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
	auto h2d_end = std::chrono::high_resolution_clock::now();
	auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>(
				h2d_end - h2d_start)
				.count();

	// Set up execution parameters
	int threads, blocks;
	auto_grid_config(h_N, threads, blocks, args.threads_per_block,
			 args.num_blocks);

	// Warm-up run
	vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();

	// Print benchmark header
	print_benchmark_header("vectorAdd", iterations, bytes, blocks, threads);

	std::vector<double> kernel_times;
	kernel_times.reserve(iterations);

	auto start_time = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iterations; i++) {
		auto kernel_start = std::chrono::high_resolution_clock::now();
		vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();
		auto kernel_end = std::chrono::high_resolution_clock::now();

		auto kernel_time =
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				kernel_end - kernel_start)
				.count() /
			1000.0;
		kernel_times.push_back(kernel_time);
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
		end_time - start_time);

	// Copy result back to host
	auto d2h_start = std::chrono::high_resolution_clock::now();
	cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
	auto d2h_end = std::chrono::high_resolution_clock::now();
	auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>(
				d2h_end - d2h_start)
				.count();

	// Print benchmark results
	double total_time_us = duration.count() / 1000.0;
	double avg_time_us = total_time_us / iterations;

	print_benchmark_results("vectorAdd", iterations, bytes, host_alloc_time,
				device_alloc_time, h2d_time, d2h_time,
				total_time_us, avg_time_us, &kernel_times);

	std::cout << "\nValidation check: C[0] = " << h_C[0]
		  << ", C[1] = " << h_C[1] << "\n";
	std::cout << "========================================\n";
	fflush(stdout);
	// Cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
