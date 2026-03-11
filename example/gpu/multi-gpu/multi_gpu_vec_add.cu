/**
 * multi_gpu_vec_add.cu - Multi-GPU vector addition example
 *
 * Demonstrates running the same CUDA kernel on multiple GPUs simultaneously.
 * When used with bpftime's GPU attach, eBPF probes are automatically
 * instrumented on all GPUs, enabling per-device tracing and profiling.
 *
 * Usage:
 *   # Compile
 *   nvcc -o multi_gpu_vec_add multi_gpu_vec_add.cu
 *
 *   # Run standalone (verify correctness)
 *   ./multi_gpu_vec_add
 *
 *   # Run with bpftime GPU attach (traces kernel execution on each GPU)
 *   bpftime load ./multi_gpu_vec_add
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
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

// Simple vector addition kernel - same kernel runs on each GPU
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		C[idx] = A[idx] + B[idx];
	}
}

struct PerGPUData {
	int device_id;
	float *d_A, *d_B, *d_C;
	std::vector<float> h_A, h_B, h_C;
	cudaStream_t stream;
	int N; // number of elements for this GPU
};

int main(int argc, char **argv)
{
	int device_count = 0;
	CUDA_CHECK(cudaGetDeviceCount(&device_count));

	if (device_count < 1) {
		fprintf(stderr, "No CUDA devices found\n");
		return 1;
	}

	// Allow user to limit the number of GPUs via command line
	int num_gpus = device_count;
	if (argc > 1) {
		num_gpus = atoi(argv[1]);
		if (num_gpus < 1 || num_gpus > device_count) {
			fprintf(stderr,
				"Invalid GPU count %d (have %d devices)\n",
				num_gpus, device_count);
			return 1;
		}
	}

	printf("Using %d GPU(s) out of %d available\n", num_gpus,
	       device_count);

	// Print device info
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
		printf("  GPU %d: %s (SM %d.%d, %d SMs)\n", i, prop.name,
		       prop.major, prop.minor,
		       prop.multiProcessorCount);
	}

	// Total problem size: 4M elements split across GPUs
	const int total_N = 4 * 1024 * 1024;
	const int per_gpu_N = total_N / num_gpus;

	// Set up per-GPU data
	std::vector<PerGPUData> gpus(num_gpus);

	for (int g = 0; g < num_gpus; g++) {
		auto &gpu = gpus[g];
		gpu.device_id = g;
		gpu.N = (g == num_gpus - 1) ? (total_N - per_gpu_N * g) :
					       per_gpu_N;

		CUDA_CHECK(cudaSetDevice(g));
		CUDA_CHECK(cudaStreamCreate(&gpu.stream));

		size_t bytes = gpu.N * sizeof(float);
		CUDA_CHECK(cudaMalloc(&gpu.d_A, bytes));
		CUDA_CHECK(cudaMalloc(&gpu.d_B, bytes));
		CUDA_CHECK(cudaMalloc(&gpu.d_C, bytes));

		// Initialize host data
		gpu.h_A.resize(gpu.N);
		gpu.h_B.resize(gpu.N);
		gpu.h_C.resize(gpu.N);

		int offset = g * per_gpu_N;
		for (int i = 0; i < gpu.N; i++) {
			gpu.h_A[i] = (float)(offset + i);
			gpu.h_B[i] = (float)(offset + i) * 2.0f;
		}

		// Copy to device (async)
		CUDA_CHECK(cudaMemcpyAsync(gpu.d_A, gpu.h_A.data(), bytes,
					   cudaMemcpyHostToDevice, gpu.stream));
		CUDA_CHECK(cudaMemcpyAsync(gpu.d_B, gpu.h_B.data(), bytes,
					   cudaMemcpyHostToDevice, gpu.stream));
	}

	// Run 5 iterations to demonstrate repeated multi-GPU kernel launches
	for (int iter = 0; iter < 5; iter++) {
		printf("\n--- Iteration %d ---\n", iter + 1);

		// Launch kernel on all GPUs concurrently
		for (int g = 0; g < num_gpus; g++) {
			auto &gpu = gpus[g];
			CUDA_CHECK(cudaSetDevice(g));

			int threads = 256;
			int blocks = (gpu.N + threads - 1) / threads;

			vectorAdd<<<blocks, threads, 0, gpu.stream>>>(
				gpu.d_A, gpu.d_B, gpu.d_C, gpu.N);
		}

		// Synchronize and verify results
		for (int g = 0; g < num_gpus; g++) {
			auto &gpu = gpus[g];
			CUDA_CHECK(cudaSetDevice(g));
			CUDA_CHECK(cudaStreamSynchronize(gpu.stream));

			size_t bytes = gpu.N * sizeof(float);
			CUDA_CHECK(cudaMemcpy(gpu.h_C.data(), gpu.d_C, bytes,
					      cudaMemcpyDeviceToHost));

			// Verify a few elements
			int offset = g * per_gpu_N;
			bool correct = true;
			for (int i = 0; i < 3 && i < gpu.N; i++) {
				float expected = (float)(offset + i) * 3.0f;
				if (abs(gpu.h_C[i] - expected) > 1e-5) {
					printf("GPU %d: MISMATCH at [%d]: got "
					       "%f, expected %f\n",
					       g, i, gpu.h_C[i], expected);
					correct = false;
				}
			}
			printf("GPU %d: %s (%d elements, C[0]=%.1f, "
			       "C[1]=%.1f)\n",
			       g, correct ? "PASS" : "FAIL", gpu.N,
			       gpu.h_C[0], gpu.N > 1 ? gpu.h_C[1] : 0.0f);
		}
	}

	// Cleanup
	for (int g = 0; g < num_gpus; g++) {
		auto &gpu = gpus[g];
		CUDA_CHECK(cudaSetDevice(g));
		CUDA_CHECK(cudaFree(gpu.d_A));
		CUDA_CHECK(cudaFree(gpu.d_B));
		CUDA_CHECK(cudaFree(gpu.d_C));
		CUDA_CHECK(cudaStreamDestroy(gpu.stream));
	}

	printf("\nMulti-GPU vector addition completed successfully.\n");
	return 0;
}
