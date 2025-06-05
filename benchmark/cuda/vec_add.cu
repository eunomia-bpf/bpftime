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
#include <chrono>

__constant__ int d_N;

// A simple vector addition kernel for benchmarking
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    // Parse iterations
    int iterations = 10000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    // Set vector size in constant memory
    const int h_N = 1 << 20;  // 1M elements
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));
    
    size_t bytes = h_N * sizeof(float);
    
    // Allocate and initialize host memory
    std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);
    
    for (int i = 0; i < h_N; ++i)
    {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    // Allocate Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Set up execution parameters
    int threads = 256;
    int blocks = (h_N + threads - 1) / threads;

    // Warm-up run
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Benchmark loop
    std::cout << "Running benchmark with " << iterations << " iterations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Print benchmark results
    double avg_time_us = duration.count() / 1000.0 / iterations;
    std::cout << "Benchmark results:\n";
    std::cout << "Total time: " << duration.count() / 1000.0 << " us\n";
    std::cout << "Average kernel time: " << avg_time_us << " us\n";
    std::cout << "Validation check: C[0] = " << h_C[0] << ", C[1] = " << h_C[1] << "\n";
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
} 