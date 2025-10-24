#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

// Statistics structure
struct BenchmarkStats {
    double mean;
    double median;
    double min;
    double max;
    double p95;
    double p99;
    double stddev;
};

// Calculate statistics from timing data
inline BenchmarkStats calculate_stats(std::vector<double>& times) {
    BenchmarkStats stats;
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

// Print benchmark header
inline void print_benchmark_header(const char* kernel_name, int iterations, size_t data_size_bytes,
                                   int blocks, int threads) {
    std::cout << "Running benchmark with " << iterations << " iterations...\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << "Data size: " << data_size_bytes / 1024 / 1024 << " MB\n";
    std::cout << "Grid config: " << blocks << " blocks × " << threads << " threads/block = "
              << blocks * threads << " total threads\n";
    std::cout << std::endl;
}

// Print setup phase timing
inline void print_setup_timing(long host_alloc_us, long device_alloc_us,
                               long h2d_us, long d2h_us, size_t bytes) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Setup phase:\n";
    std::cout << "  Host allocation:   " << host_alloc_us << " us\n";
    std::cout << "  Device allocation: " << device_alloc_us << " us\n";
    std::cout << "  Host to device:    " << h2d_us << " us ("
              << (bytes / 1024.0 / 1024.0) / (h2d_us / 1000000.0) << " GB/s)\n";
    std::cout << "  Device to host:    " << d2h_us << " us ("
              << (bytes / 1024.0 / 1024.0) / (d2h_us / 1000000.0) << " GB/s)\n\n";
}

// Print kernel execution results
inline void print_kernel_results(double total_time_us, double avg_time_us, int iterations,
                                 std::vector<double>* kernel_times = nullptr) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Kernel execution:\n";
    std::cout << "  Total time:        " << total_time_us << " us\n";
    std::cout << "  Average kernel time: " << avg_time_us << " us\n";
    std::cout << "  Throughput:        " << (iterations * 1000000.0) / total_time_us
              << " kernels/sec\n";

    if (kernel_times && !kernel_times->empty()) {
        BenchmarkStats stats = calculate_stats(*kernel_times);
        std::cout << "\nDetailed statistics (μs):\n";
        std::cout << "  Mean:              " << stats.mean << " us\n";
        std::cout << "  Median:            " << stats.median << " us\n";
        std::cout << "  Min:               " << stats.min << " us\n";
        std::cout << "  Max:               " << stats.max << " us\n";
        std::cout << "  P95:               " << stats.p95 << " us\n";
        std::cout << "  P99:               " << stats.p99 << " us\n";
        std::cout << "  Std dev:           " << stats.stddev << " us\n";
    }
}

// Print complete benchmark results
inline void print_benchmark_results(const char* kernel_name, int iterations, size_t bytes,
                                    long host_alloc_us, long device_alloc_us,
                                    long h2d_us, long d2h_us,
                                    double total_time_us, double avg_time_us,
                                    std::vector<double>* kernel_times = nullptr) {
    std::cout << "========================================\n";
    std::cout << "Benchmark results:\n";
    std::cout << "========================================\n";
    print_setup_timing(host_alloc_us, device_alloc_us, h2d_us, d2h_us, bytes);
    print_kernel_results(total_time_us, avg_time_us, iterations, kernel_times);
}

// Argument parser structure
struct BenchmarkArgs {
    int size;                  // Problem size (elements, matrix dim, etc.)
    int iterations;            // Number of iterations
    int threads_per_block;     // Threads per block (0 = auto)
    int num_blocks;            // Number of blocks (0 = auto)
    int block_size;            // Block size for matrix operations

    // Parse command line arguments
    // Usage: program [size] [iterations] [threads_per_block] [num_blocks]
    static BenchmarkArgs parse(int argc, char** argv,
                              int default_size = 1024,
                              int default_iterations = 10,
                              int default_block_size = 16) {
        BenchmarkArgs args;
        args.size = (argc > 1) ? atoi(argv[1]) : default_size;
        args.iterations = (argc > 2) ? atoi(argv[2]) : default_iterations;
        args.threads_per_block = (argc > 3) ? atoi(argv[3]) : 0;
        args.num_blocks = (argc > 4) ? atoi(argv[4]) : 0;
        args.block_size = default_block_size;
        return args;
    }
};

// Auto-calculate grid configuration
inline void auto_grid_config(int problem_size, int& threads, int& blocks,
                             int threads_per_block = 0, int num_blocks = 0) {
    if (threads_per_block > 0 && num_blocks > 0) {
        threads = threads_per_block;
        blocks = num_blocks;
    } else if (threads_per_block > 0) {
        threads = threads_per_block;
        blocks = (problem_size + threads - 1) / threads;
    } else if (num_blocks > 0) {
        blocks = num_blocks;
        threads = (problem_size + blocks - 1) / blocks;
        threads = ((threads + 31) / 32) * 32;  // Round to warp size
    } else {
        // Auto mode
        if (problem_size <= 32) {
            threads = 32;
        } else if (problem_size <= 128) {
            threads = 128;
        } else if (problem_size <= 1024) {
            threads = 256;
        } else {
            threads = 512;
        }
        blocks = (problem_size + threads - 1) / threads;
    }
}

#endif // BENCHMARK_COMMON_H
