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
#include <numeric>
#include <algorithm>

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

// Helper function to calculate statistics
struct Stats {
    double mean;
    double median;
    double min;
    double max;
    double p95;
    double p99;
    double stddev;
};

Stats calculate_stats(std::vector<double>& times) {
    Stats stats;
    std::sort(times.begin(), times.end());

    stats.min = times.front();
    stats.max = times.back();
    stats.median = times[times.size() / 2];
    stats.p95 = times[static_cast<size_t>(times.size() * 0.95)];
    stats.p99 = times[static_cast<size_t>(times.size() * 0.99)];

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    stats.mean = sum / times.size();

    double sq_sum = 0.0;
    for (auto t : times) {
        sq_sum += (t - stats.mean) * (t - stats.mean);
    }
    stats.stddev = std::sqrt(sq_sum / times.size());

    return stats;
}

int main(int argc, char **argv)
{
    // Parse iterations
    int iterations = 10000;
    bool detailed = false;

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    if (argc > 2 && std::string(argv[2]) == "--detailed") {
        detailed = true;
    }

    // Set vector size in constant memory
    const int h_N = 1 << 20;  // 1M elements
    cudaError_t err = cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    size_t bytes = h_N * sizeof(float);

    // Allocate and initialize host memory
    auto host_alloc_start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);

    for (int i = 0; i < h_N; ++i)
    {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }
    auto host_alloc_end = std::chrono::high_resolution_clock::now();
    auto host_alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(
        host_alloc_end - host_alloc_start).count();

    // Allocate Device memory
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
    int threads = 256;
    int blocks = (h_N + threads - 1) / threads;

    // Warm-up run
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Get CUDA device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Benchmark loop with detailed timing
    std::cout << "Running benchmark with " << iterations << " iterations...\n";
    std::cout << "Configuration: " << blocks << " blocks × " << threads << " threads = "
              << blocks * threads << " threads\n";
    std::cout << "Vector size: " << h_N << " elements (" << bytes / 1024 / 1024 << " MB)\n\n";

    std::vector<double> kernel_times;
    if (detailed) {
        kernel_times.reserve(iterations);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto kernel_start = std::chrono::high_resolution_clock::now();
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        auto kernel_end = std::chrono::high_resolution_clock::now();

        if (detailed) {
            auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                kernel_end - kernel_start).count() / 1000.0;
            kernel_times.push_back(kernel_time);
        }
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
    double avg_time_us = duration.count() / 1000.0 / iterations;
    std::cout << "========================================\n";
    std::cout << "Benchmark results:\n";
    std::cout << "========================================\n";
    std::cout << "Setup phase:\n";
    std::cout << "  Host allocation:   " << host_alloc_time << " us\n";
    std::cout << "  Device allocation: " << device_alloc_time << " us\n";
    std::cout << "  Host to device:    " << h2d_time << " us ("
              << (bytes / 1024.0 / 1024.0) / (h2d_time / 1000000.0) << " GB/s)\n";
    std::cout << "  Device to host:    " << d2h_time << " us ("
              << (bytes / 1024.0 / 1024.0) / (d2h_time / 1000000.0) << " GB/s)\n\n";

    std::cout << "Kernel execution:\n";
    std::cout << "  Total time:        " << duration.count() / 1000.0 << " us\n";
    std::cout << "  Average kernel time: " << avg_time_us << " us\n";
    std::cout << "  Throughput:        " << (iterations * 1000000.0) / (duration.count() / 1000.0)
              << " kernels/sec\n";

    if (detailed && !kernel_times.empty()) {
        Stats stats = calculate_stats(kernel_times);
        std::cout << "\nDetailed statistics (μs):\n";
        std::cout << "  Mean:              " << stats.mean << " us\n";
        std::cout << "  Median:            " << stats.median << " us\n";
        std::cout << "  Min:               " << stats.min << " us\n";
        std::cout << "  Max:               " << stats.max << " us\n";
        std::cout << "  P95:               " << stats.p95 << " us\n";
        std::cout << "  P99:               " << stats.p99 << " us\n";
        std::cout << "  Std dev:           " << stats.stddev << " us\n";
    }

    std::cout << "\nValidation check: C[0] = " << h_C[0] << ", C[1] = " << h_C[1] << "\n";
    std::cout << "========================================\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
} 