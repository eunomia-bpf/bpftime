// thread_time_distribution_demo.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple CUDA error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",               \
                    cudaGetErrorString(err__), err__, __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Per-thread timing info
struct ThreadTime {
    unsigned int blockId;     // block index (1D grid)
    unsigned int threadId;    // global thread ID (1D)
    unsigned long long cycles; // cycles spent in work section
};

// Kernel: each thread runs a synthetic workload, and we measure its time
__global__ void timed_work_kernel(ThreadTime* out, int base_iters)
{
    const unsigned int blockId = blockIdx.x;
    const unsigned int threadIdInBlock = threadIdx.x;
    const unsigned int globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread runs a slightly different number of iterations
    // to produce a non-trivial distribution.
    int my_iters = base_iters + (globalThreadId & 0x1F); // vary per thread

    // Time the "work" section using clock64()
    unsigned long long start = clock64();

    volatile float acc = 0.0f;
    for (int i = 0; i < my_iters; ++i) {
        acc += 1.0f;  // trivial arithmetic
    }

    unsigned long long end = clock64();

    // Prevent the compiler from optimizing out the loop completely
    if (acc == -1.0f) {
        printf("This will never be printed\n");
    }

    ThreadTime t;
    t.blockId  = blockId;
    t.threadId = globalThreadId;
    t.cycles   = end - start;

    out[globalThreadId] = t;
}

int main()
{
    // Again, keep it small for printing
    const int BLOCKS            = 4;
    const int THREADS_PER_BLOCK = 64;
    const int TOTAL_THREADS     = BLOCKS * THREADS_PER_BLOCK;

    const int BASE_ITERS        = 100000; // base workload per thread

    printf("=== Per-thread runtime distribution demo ===\n");
    printf("Grid: %d blocks, Block: %d threads (total %d threads)\n",
           BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);

    // Allocate device and host buffers
    ThreadTime* d_times = nullptr;
    CHECK_CUDA(cudaMalloc(&d_times, TOTAL_THREADS * sizeof(ThreadTime)));

    ThreadTime* h_times = (ThreadTime*)std::malloc(TOTAL_THREADS * sizeof(ThreadTime));
    if (!h_times) {
        fprintf(stderr, "Host malloc failed\n");
        std::exit(EXIT_FAILURE);
    }

    // Launch kernel
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);

    timed_work_kernel<<<grid, block>>>(d_times, BASE_ITERS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_times, d_times,
                          TOTAL_THREADS * sizeof(ThreadTime),
                          cudaMemcpyDeviceToHost));

    // Get device clock rate to convert cycles -> time
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // clockRate: kHz (cycles per millisecond)
    const double sm_freq_hz = (double)prop.clockRate * 1000.0;

    printf("\nDevice: %s\n", prop.name);
    printf("SM clock rate: %.3f MHz\n\n", sm_freq_hz / 1.0e6);

    // Print per-thread timing
    printf("Per-thread timing:\n");
    printf("tid  block     cycles        time_us\n");
    printf("--------------------------------------\n");

    unsigned long long min_cycles = ~0ULL;
    unsigned long long max_cycles = 0;
    unsigned long long sum_cycles = 0;

    for (int i = 0; i < TOTAL_THREADS; ++i) {
        const ThreadTime& t = h_times[i];

        double time_sec = (double)t.cycles / sm_freq_hz;
        double time_us  = time_sec * 1.0e6;

        printf("%3u  %5u  %10llu  %10.3f\n",
               t.threadId, t.blockId,
               (unsigned long long)t.cycles, time_us);

        if (t.cycles < min_cycles) min_cycles = t.cycles;
        if (t.cycles > max_cycles) max_cycles = t.cycles;
        sum_cycles += t.cycles;
    }

    double avg_cycles = (double)sum_cycles / TOTAL_THREADS;
    double min_us = (double)min_cycles / sm_freq_hz * 1.0e6;
    double max_us = (double)max_cycles / sm_freq_hz * 1.0e6;
    double avg_us = avg_cycles / sm_freq_hz * 1.0e6;

    printf("\nSummary over all threads:\n");
    printf("  min cycles = %llu (%.3f us)\n",
           (unsigned long long)min_cycles, min_us);
    printf("  max cycles = %llu (%.3f us)\n",
           (unsigned long long)max_cycles, max_us);
    printf("  avg cycles = %.1f (%.3f us)\n",
           avg_cycles, avg_us);

    // Cleanup
    CHECK_CUDA(cudaFree(d_times));
    std::free(h_times);

    return 0;
}