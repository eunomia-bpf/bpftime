#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
	do {                                                                 \
			cudaError_t err__ = (call);                                  \
			if (err__ != cudaSuccess) {                                 \
			std::cerr << "CUDA error at " << __FILE__ << ":" \
				  << __LINE__ << ": "                    \
				  << cudaGetErrorString(err__) << "\n";  \
			std::exit(1);                                       \
		}                                                            \
	} while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[idx] = A[idx] + B[idx];
}

int main()
{
	const int h_N = 1 << 20;

	size_t bytes = h_N * sizeof(float);
	std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);
	for (int i = 0; i < h_N; ++i) {
		h_A[i] = float(i);
		h_B[i] = float(2 * i);
	}

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	CUDA_CHECK(cudaMalloc(&d_A, bytes));
	CUDA_CHECK(cudaMalloc(&d_B, bytes));
	CUDA_CHECK(cudaMalloc(&d_C, bytes));
	CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

	int threads = 256;
	int blocks = (h_N + threads - 1) / threads;

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	// Baseline launch (non-graph) to verify the kernel runs and to provide
	// a reference point when comparing against graph-launched behavior.
	vectorAdd<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C);
	CUDA_CHECK(cudaStreamSynchronize(stream));

	cudaGraph_t graph;
	CUDA_CHECK(cudaGraphCreate(&graph, 0));

	cudaGraphNode_t kernel_node;
	cudaKernelNodeParams node_params{};
	void *kernel_args[] = { &d_A, &d_B, &d_C };
	node_params.func = (void *)vectorAdd;
	node_params.gridDim = dim3(blocks, 1, 1);
	node_params.blockDim = dim3(threads, 1, 1);
	node_params.sharedMemBytes = 0;
	node_params.kernelParams = kernel_args;
	node_params.extra = nullptr;

	CUDA_CHECK(cudaGraphAddKernelNode(&kernel_node, graph, nullptr, 0,
					  &node_params));

	// Trigger KernelNodeSetParams hooks (no-op update).
	CUDA_CHECK(cudaGraphKernelNodeSetParams(kernel_node, &node_params));

	cudaGraphExec_t graph_exec;
	CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

	// Trigger ExecKernelNodeSetParams hooks (no-op update).
	CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph_exec, kernel_node,
						    &node_params));

		// Default: keep running to make it easy to observe repeated
		// cudaGraphLaunch behavior from bpftime tooling. Set a positive
		// BPFTIME_CUDAGRAPH_ITERS to run a finite number of iterations.
		int iters = 0;
		if (const char *env = std::getenv("BPFTIME_CUDAGRAPH_ITERS");
		    env != nullptr && env[0] != '\0') {
			iters = std::atoi(env);
		}

		for (int iter = 0; iters <= 0 || iter < iters; iter++) {
			CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytes, stream));
			CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));

			CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes,
					      cudaMemcpyDeviceToHost));

			// Use stderr for immediate flushing under tooling/timeouts.
			std::fprintf(stderr, "C[0]=%.1f (expected 0), C[1]=%.1f (expected 3)\n",
				     h_C[0], h_C[1]);
			sleep(1);
		}

	CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
	CUDA_CHECK(cudaGraphDestroy(graph));
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	return 0;
}
