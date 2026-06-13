/**
 * multi_gpu_vec_add.cu - Multi-GPU load balance monitor workload
 *
 * Distributes intentionally IMBALANCED workloads across GPUs to demonstrate
 * load balance monitoring. Each GPU gets progressively more work:
 *   GPU 0: 1x base_N elements
 *   GPU 1: 2x base_N elements
 *   GPU k: (k+1)x base_N elements
 *
 * The program uses CUDA events for host-side per-GPU timing and prints
 * a load balance dashboard each iteration. When used with bpftime's GPU
 * attach, eBPF probes independently measure GPU-internal per-block timing.
 *
 * Usage:
 *   nvcc -cudart shared -o multi_gpu_vec_add multi_gpu_vec_add.cu
 *   ./multi_gpu_vec_add [num_gpus] [iterations]
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
	do {                                                                   \
		cudaError_t err = call;                                        \
		if (err != cudaSuccess) {                                      \
			fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
				__FILE__, __LINE__,                            \
				cudaGetErrorString(err));                       \
			exit(1);                                               \
		}                                                              \
	} while (0)

// Compute-bound kernel: vector addition with extra arithmetic to make
// execution time measurable and proportional to N.
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		float a = A[idx], b = B[idx];
		// Extra arithmetic to amplify per-element cost
		float r = a + b;
		for (int k = 0; k < 64; k++) {
			r = r * 1.00001f + 0.00001f;
		}
		C[idx] = r;
	}
}

struct PerGPUData {
	int device_id;
	float *d_A, *d_B, *d_C;
	std::vector<float> h_A, h_B, h_C;
	cudaStream_t stream;
	cudaEvent_t evt_start, evt_stop;
	int N;
	float last_ms; // last measured kernel time in ms
};

static void print_balance_dashboard(const std::vector<PerGPUData> &gpus,
				    int iter)
{
	int num = (int)gpus.size();
	float min_ms = FLT_MAX, max_ms = 0, sum_ms = 0;
	for (int g = 0; g < num; g++) {
		float ms = gpus[g].last_ms;
		if (ms < min_ms)
			min_ms = ms;
		if (ms > max_ms)
			max_ms = ms;
		sum_ms += ms;
	}
	float avg_ms = sum_ms / num;
	float variance = 0;
	for (int g = 0; g < num; g++) {
		float d = gpus[g].last_ms - avg_ms;
		variance += d * d;
	}
	float stdev_ms = sqrtf(variance / num);
	float imbalance = (max_ms > 0) ? (max_ms - min_ms) / max_ms * 100.0f :
					  0;
	// Utilization: ideal total time = max_ms, actual idle =
	// sum(max_ms-gpu_ms)
	float util =
		(max_ms > 0) ? sum_ms / (max_ms * num) * 100.0f : 100.0f;

	printf("\n╔══════════════════════════════════════════════════"
	       "═══════════╗\n");
	printf("║  MULTI-GPU LOAD BALANCE DASHBOARD  -  Iteration "
	       "%-4d        ║\n",
	       iter);
	printf("╠══════════════════════════════════════════════════"
	       "═══════════╣\n");
	printf("║  GPU │  Elements  │  Time (ms)  │  "
	       "Relative  │  Bar       ║\n");
	printf("╠══════════════════════════════════════════════════"
	       "═══════════╣\n");
	for (int g = 0; g < num; g++) {
		float ms = gpus[g].last_ms;
		float rel = (max_ms > 0) ? ms / max_ms : 0;
		int bar_len = (int)(rel * 10);

		char bar[12];
		for (int i = 0; i < 10; i++)
			bar[i] = (i < bar_len) ? '#' : '.';
		bar[10] = '\0';

		printf("║  %2d  │ %9d  │  %9.3f  │   %5.1f%%   │ "
		       " %s ║\n",
		       g, gpus[g].N, ms, rel * 100, bar);
	}
	printf("╠══════════════════════════════════════════════════"
	       "═══════════╣\n");
	printf("║  Min: %7.3f ms   Max: %7.3f ms   Avg: %7.3f "
	       "ms        ║\n",
	       min_ms, max_ms, avg_ms);
	printf("║  Stdev: %5.3f ms   Imbalance: %5.1f%%   "
	       "Utilization: %5.1f%%   ║\n",
	       stdev_ms, imbalance, util);
	printf("╚══════════════════════════════════════════════════"
	       "═══════════╝\n");
}

int main(int argc, char **argv)
{
	int device_count = 0;
	CUDA_CHECK(cudaGetDeviceCount(&device_count));
	if (device_count < 1) {
		fprintf(stderr, "No CUDA devices found\n");
		return 1;
	}

	int num_gpus = (device_count > 4) ? 4 : device_count;
	int iterations = 10;
	if (argc > 1)
		num_gpus = atoi(argv[1]);
	if (argc > 2)
		iterations = atoi(argv[2]);
	if (num_gpus < 1 || num_gpus > device_count) {
		fprintf(stderr, "Invalid GPU count %d (have %d)\n", num_gpus,
			device_count);
		return 1;
	}

	printf("Multi-GPU Load Balance Monitor\n");
	printf("GPUs: %d  Iterations: %d\n\n", num_gpus, iterations);

	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
		printf("  GPU %d: %s (SM %d.%d, %d SMs)\n", i, prop.name,
		       prop.major, prop.minor,
		       prop.multiProcessorCount);
	}

	// Imbalanced workload: GPU k gets (k+1) * base_N elements
	const int base_N = 512 * 1024; // 512K base

	std::vector<PerGPUData> gpus(num_gpus);

	for (int g = 0; g < num_gpus; g++) {
		auto &gpu = gpus[g];
		gpu.device_id = g;
		gpu.N = base_N * (g + 1); // Intentional imbalance!
		gpu.last_ms = 0;

		CUDA_CHECK(cudaSetDevice(g));
		CUDA_CHECK(cudaStreamCreate(&gpu.stream));
		CUDA_CHECK(cudaEventCreate(&gpu.evt_start));
		CUDA_CHECK(cudaEventCreate(&gpu.evt_stop));

		size_t bytes = (size_t)gpu.N * sizeof(float);
		CUDA_CHECK(cudaMalloc(&gpu.d_A, bytes));
		CUDA_CHECK(cudaMalloc(&gpu.d_B, bytes));
		CUDA_CHECK(cudaMalloc(&gpu.d_C, bytes));

		gpu.h_A.resize(gpu.N);
		gpu.h_B.resize(gpu.N);
		gpu.h_C.resize(gpu.N, 0);
		for (int i = 0; i < gpu.N; i++) {
			gpu.h_A[i] = (float)(i % 1000) * 0.001f;
			gpu.h_B[i] = (float)(i % 1000) * 0.002f;
		}
		CUDA_CHECK(cudaMemcpyAsync(gpu.d_A, gpu.h_A.data(), bytes,
					   cudaMemcpyHostToDevice, gpu.stream));
		CUDA_CHECK(cudaMemcpyAsync(gpu.d_B, gpu.h_B.data(), bytes,
					   cudaMemcpyHostToDevice, gpu.stream));
	}

	printf("\nWorkload distribution (intentionally imbalanced):\n");
	for (int g = 0; g < num_gpus; g++)
		printf("  GPU %d: %d elements (%.1fx base)\n", g, gpus[g].N,
		       (float)(g + 1));

	for (int iter = 1; iter <= iterations; iter++) {
		// Launch all GPUs with CUDA event timing
		for (int g = 0; g < num_gpus; g++) {
			auto &gpu = gpus[g];
			CUDA_CHECK(cudaSetDevice(g));

			int threads = 256;
			int blocks = (gpu.N + threads - 1) / threads;

			CUDA_CHECK(
				cudaEventRecord(gpu.evt_start, gpu.stream));
			vectorAdd<<<blocks, threads, 0, gpu.stream>>>(
				gpu.d_A, gpu.d_B, gpu.d_C, gpu.N);
			CUDA_CHECK(
				cudaEventRecord(gpu.evt_stop, gpu.stream));
		}

		// Synchronize and read timing
		for (int g = 0; g < num_gpus; g++) {
			auto &gpu = gpus[g];
			CUDA_CHECK(cudaSetDevice(g));
			CUDA_CHECK(cudaEventSynchronize(gpu.evt_stop));
			CUDA_CHECK(cudaEventElapsedTime(&gpu.last_ms,
							gpu.evt_start,
							gpu.evt_stop));
		}

		print_balance_dashboard(gpus, iter);

		if (iter < iterations)
			sleep(1);
	}

	// Cleanup
	for (int g = 0; g < num_gpus; g++) {
		CUDA_CHECK(cudaSetDevice(g));
		CUDA_CHECK(cudaFree(gpus[g].d_A));
		CUDA_CHECK(cudaFree(gpus[g].d_B));
		CUDA_CHECK(cudaFree(gpus[g].d_C));
		CUDA_CHECK(cudaEventDestroy(gpus[g].evt_start));
		CUDA_CHECK(cudaEventDestroy(gpus[g].evt_stop));
		CUDA_CHECK(cudaStreamDestroy(gpus[g].stream));
	}

	printf("\nDone.\n");
	return 0;
}
