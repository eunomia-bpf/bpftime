#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include "benchmark_common.h"

#define BLOCK_SIZE 16

// Matrix multiplication kernel with shared memory tiling
template <int BS>
__global__ void vectorAdd(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BS * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BS;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BS * bx;
    int bStep = BS * wB;

    // Accumulated value
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BS][BS];
        __shared__ float Bs[BS][BS];

        // Load matrices from global memory to shared memory
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        // Multiply the two matrices together
#pragma unroll
        for (int k = 0; k < BS; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the block sub-matrix to device memory
    int c = wB * BS * by + BS * bx;
    C[c + wB * ty + tx] = Csub;
}

int main(int argc, char **argv)
{
    // Parse command line arguments: matrix_dim, iterations, (threads_per_block not used for matrix mul)
    BenchmarkArgs args = BenchmarkArgs::parse(argc, argv, 320, 10, BLOCK_SIZE);
    int matrix_dim = args.size;  // Square matrix dimension
    int iterations = args.iterations;
    int block_size = (args.block_size > 0) ? args.block_size : BLOCK_SIZE;

    // Ensure matrix dimensions are multiples of block size
    if (matrix_dim % block_size != 0) {
        matrix_dim = ((matrix_dim + block_size - 1) / block_size) * block_size;
    }

    size_t matrix_size = matrix_dim * matrix_dim;
    size_t bytes = matrix_size * sizeof(float);

    // Allocate and initialize host memory
    auto host_alloc_start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_A(matrix_size), h_B(matrix_size), h_C(matrix_size);

    for (size_t i = 0; i < matrix_size; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 0.01f;
    }
    auto host_alloc_end = std::chrono::high_resolution_clock::now();
    auto host_alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(
        host_alloc_end - host_alloc_start).count();

    // Allocate device memory
    auto device_alloc_start = std::chrono::high_resolution_clock::now();
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    auto device_alloc_end = std::chrono::high_resolution_clock::now();
    auto device_alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(
        device_alloc_end - device_alloc_start).count();

    // Copy to device
    auto h2d_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    auto h2d_end = std::chrono::high_resolution_clock::now();
    auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>(
        h2d_end - h2d_start).count();

    // Set up execution parameters
    dim3 threads(block_size, block_size);
    dim3 blocks(matrix_dim / threads.x, matrix_dim / threads.y);

    // Warm-up run
    if (block_size == 16) {
        vectorAdd<16><<<blocks, threads>>>(d_C, d_A, d_B, matrix_dim, matrix_dim);
    } else {
        vectorAdd<32><<<blocks, threads>>>(d_C, d_A, d_B, matrix_dim, matrix_dim);
    }
    cudaDeviceSynchronize();

    // Print benchmark header
    print_benchmark_header("MatrixMul", iterations, bytes, blocks.x * blocks.y, threads.x * threads.y);
    std::cout << "Matrix: " << matrix_dim << "x" << matrix_dim << "\n\n";

    std::vector<double> kernel_times;
    kernel_times.reserve(iterations);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto kernel_start = std::chrono::high_resolution_clock::now();

        if (block_size == 16) {
            vectorAdd<16><<<blocks, threads>>>(d_C, d_A, d_B, matrix_dim, matrix_dim);
        } else {
            vectorAdd<32><<<blocks, threads>>>(d_C, d_A, d_B, matrix_dim, matrix_dim);
        }

        cudaDeviceSynchronize();
        auto kernel_end = std::chrono::high_resolution_clock::now();

        auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            kernel_end - kernel_start).count() / 1000.0;
        kernel_times.push_back(kernel_time);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    // Copy result back to host
    auto d2h_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::high_resolution_clock::now();
    auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>(
        d2h_end - d2h_start).count();

    // Print benchmark results
    double total_time_us = duration.count() / 1000.0;
    double avg_time_us = total_time_us / iterations;

    print_benchmark_results("MatrixMul", iterations, bytes * 2,  // A and B
                           host_alloc_time, device_alloc_time,
                           h2d_time, d2h_time,
                           total_time_us, avg_time_us,
                           &kernel_times);

    // Calculate GFLOPS
    double flops = 2.0 * static_cast<double>(matrix_dim) * matrix_dim * matrix_dim;
    double gflops = (flops * 1.0e-9) / (avg_time_us / 1.0e6);
    std::cout << "\nPerformance: " << gflops << " GFLOPS\n";

    // Validation
    float expected = matrix_dim * 0.01f;  // Each row has matrix_dim * 0.01
    std::cout << "Validation: C[0] = " << h_C[0] << ", expected ~" << expected << "\n";
    std::cout << "========================================\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
