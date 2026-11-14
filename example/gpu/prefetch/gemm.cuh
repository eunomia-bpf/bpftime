/**
 * gemm.cuh - GEMM kernel from PolyBench/GPU
 *
 * Source: PolyBench/GPU 1.0
 * Original file: CUDA/GEMM/gemm.cu
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Web: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * This is an adapted standalone version for UVM benchmark integration.
 * The kernel computes: C = alpha * A * B + beta * C
 */

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

// ============================================================================
// Original PolyBench GEMM Kernel (adapted for standalone use)
// ============================================================================
// C = alpha * A * B + beta * C
// A: NI x NK, B: NK x NJ, C: NI x NJ

__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha,
			    DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b,
			    DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	// Prefetch the working set into L2 at the start of the kernel.
	if ((i < ni) && (j < nj)) {
		// Prefetch row i of A into L2.
#pragma unroll 4
		for (int k = 0; k < nk; k += 8) {
			prefetch_l2(&a[i * nk + k]);
		}

		// Prefetch column j of B into L2.
#pragma unroll 4
		for (int k = 0; k < nk; k += 8) {
			prefetch_l2(&b[k * nj + j]);
		}

		// Prefetch the corresponding C element.
		prefetch_l2(&c[i * nj + j]);
	}

	if ((i < ni) && (j < nj)) {
		DATA_TYPE c_val = c[i * nj + j] * beta;

		for (int k = 0; k < nk; k++) {
			c_val += alpha * a[i * nk + k] * b[k * nj + j];
		}

		c[i * nj + j] = c_val;
	}
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

inline void run_gemm(size_t total_working_set, const std::string &mode,
		     size_t stride_bytes, int iterations,
		     std::vector<float> &runtimes, KernelResult &result)
{
	(void)stride_bytes;

	// =========================================================================
	// LLM-style weight reuse design.
	// - Keep each layer at a fixed practical size (~180 MB/layer).
	// - Scale total work by changing the layer count.
	// - Traverse all layers for multiple tokens to create temporal
	// locality.
	// =========================================================================

	// LLaMA-7B-like dimensions: dim=4096, hidden=11008.
	// Per-layer weights: 4096 * 11008 * 4 bytes, about 180 MB.
	const int dim = 4096;
	const int hidden = 11008;
	size_t layer_size = (size_t)dim * hidden * sizeof(DATA_TYPE); // ~180MB
								      // per
								      // layer

	// Scale the number of layers instead of increasing a single layer.
	int num_layers = total_working_set / layer_size;
	if (num_layers < 1)
		num_layers = 1;
	if (num_layers > 200)
		num_layers = 200; // Keep runtime bounded.

	size_t weights_size = (size_t)num_layers * layer_size;

	// Allocate the weight buffer, which is the dominant memory footprint.
	DATA_TYPE *weights;
	if (mode == "device") {
		CUDA_CHECK(cudaMalloc(&weights, weights_size));
		CUDA_CHECK(cudaMemset(weights, 0, weights_size));
	} else {
		CUDA_CHECK(cudaMallocManaged(&weights, weights_size));
		// Initialize on the CPU so UVM pages start resident on the CPU.
		size_t total_elements = (size_t)num_layers * dim * hidden;
		size_t report_interval = total_elements / 20; // 5% increments
		if (report_interval == 0)
			report_interval = 1;
		int last_percent = -1;

		fprintf(stderr, "Initializing weights (%zu MB)...\n",
			weights_size / (1024 * 1024));
		for (size_t i = 0; i < total_elements; i++) {
			weights[i] = (DATA_TYPE)(i % 1000) / 1000.0f;
			if (i % report_interval == 0) {
				int percent = (int)(i * 100 / total_elements);
				if (percent != last_percent &&
				    percent % 5 == 0) {
					// fprintf(stderr, "  %d%% complete\r",
					// percent); fflush(stderr);
					last_percent = percent;
				}
			}
		}
		fprintf(stderr, "  100%% complete\n");
	}

	// Activation buffers are small and not the source of memory pressure.
	DATA_TYPE *x, *out;
	size_t x_size = dim * sizeof(DATA_TYPE);
	size_t out_size = hidden * sizeof(DATA_TYPE);

	if (mode == "device") {
		CUDA_CHECK(cudaMalloc(&x, x_size));
		CUDA_CHECK(cudaMalloc(&out, out_size));
		CUDA_CHECK(cudaMemset(x, 0, x_size));
		CUDA_CHECK(cudaMemset(out, 0, out_size));
	} else {
		CUDA_CHECK(cudaMallocManaged(&x, x_size));
		CUDA_CHECK(cudaMallocManaged(&out, out_size));
		for (int i = 0; i < dim; i++) {
			x[i] = (DATA_TYPE)i / dim;
		}
		for (int i = 0; i < hidden; i++) {
			out[i] = 0.0f;
		}
	}

	// Apply UVM hints
	if (mode != "device" && mode != "uvm") {
		int dev;
		CUDA_CHECK(cudaGetDevice(&dev));
		apply_uvm_hints(weights, weights_size, mode, dev);
		apply_uvm_hints(x, x_size, mode, dev);
		apply_uvm_hints(out, out_size, mode, dev);
		if (mode == "uvm_prefetch") {
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

	// Launch configuration for GEMV: W[dim x hidden] @ x[dim] ->
	// out[hidden], effectively (1 x dim) @ (dim x hidden).
	dim3 block(256);
	dim3 grid((hidden + block.x - 1) / block.x);

	// Traverse all layers for multiple tokens.
	int num_tokens = 10;

	fprintf(stderr,
		"GEMM config: dim=%d, hidden=%d, layers=%d, tokens=%d\n", dim,
		hidden, num_layers, num_tokens);
	fprintf(stderr, "  Layer size: %.1f MB, Total weights: %.1f MB\n",
		layer_size / (1024.0 * 1024.0),
		weights_size / (1024.0 * 1024.0));

	auto launch = [&]() {
		for (int t = 0; t < num_tokens; t++) {
			for (int l = 0; l < num_layers; l++) {
				DATA_TYPE *W =
					weights + (size_t)l * dim * hidden;
				// GEMM: (1 x dim) @ (dim x hidden) = (1 x
				// hidden) ni=1, nj=hidden, nk=dim
				gemm_kernel<<<grid, block>>>(
					1, hidden, dim, 1.0f, 0.0f, x, W, out);
				// Continue to the next layer without
				// synchronizing.
			}
		}
	};

	time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

	// bytes_accessed includes all layer weights per token. Activations are
	// small enough to ignore for this memory-pressure benchmark.
	result.bytes_accessed = (size_t)num_tokens * num_layers * dim * hidden *
				sizeof(DATA_TYPE);

	// Cleanup
	CUDA_CHECK(cudaFree(weights));
	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(out));
}

#endif // GEMM_CUH
