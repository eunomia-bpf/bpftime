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

// Change from __constant__ to regular device variable
__device__ int d_N;

// A simple vector addition kernel for benchmarking
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
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

    // Set vector size
    const int h_N = 1 << 20;  // 1M elements
    
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
    cudaError_t err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_A: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return -1;
    }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }

    // Copy to device
    err = cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }
    err = cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Set up execution parameters
    int threads = 256;
    int blocks = (h_N + threads - 1) / threads;

    // Warm-up run
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, h_N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Warm-up run failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    // Benchmark loop
    std::cout << "Running benchmark with " << iterations << " iterations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, h_N);
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    // Copy result back to host
    err = cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy results back to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }
    
    // Print benchmark results
    double avg_time_us = duration.count() / 1000.0 / iterations;
    std::cout << "Benchmark results:\n";
    std::cout << "Total time: " << duration.count() / 1000.0 << " us\n";
    std::cout << "Average kernel time: " << avg_time_us << " us\n";
    std::cout << "Validation check: C[0] = " << h_C[0] << ", C[1] = " << h_C[1] << "\n";
    
    // Cleanup
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free d_A: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free d_B: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free d_C: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
} 