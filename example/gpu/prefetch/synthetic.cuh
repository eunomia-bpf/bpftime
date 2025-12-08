#ifndef SYNTHETIC_CUH
#define SYNTHETIC_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <functional>

__device__ __forceinline__ void prefetch_l2(const void *addr)
{
	asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
}

#define CUDA_CHECK(call)                                                       \
	do {                                                                   \
		cudaError_t err = call;                                        \
		if (err != cudaSuccess) {                                      \
			throw std::runtime_error(std::string("CUDA error: ") + \
						 cudaGetErrorString(err));     \
		}                                                              \
	} while (0)

// Result structure to return performance metrics
struct KernelResult {
	size_t bytes_accessed; // Total bytes logically accessed
	float median_ms;
	float min_ms;
	float max_ms;
};

// ============================================================================
// UVM Memory Advise Helper
// ============================================================================

inline void apply_uvm_hints(void *ptr, size_t bytes, const std::string &mode,
			    int dev)
{
	if (mode == "uvm_prefetch") {
		CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, dev, 0));
	} else if (mode == "uvm_advise_read") {
		// Set read-mostly: creates read-only copies across processors
		CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetReadMostly,
					 dev));
	} else if (mode == "uvm_advise_pref_gpu") {
		// Set preferred location to GPU
		CUDA_CHECK(cudaMemAdvise(
			ptr, bytes, cudaMemAdviseSetPreferredLocation, dev));
	} else if (mode == "uvm_advise_pref_cpu") {
		// Set preferred location to CPU
		CUDA_CHECK(cudaMemAdvise(ptr, bytes,
					 cudaMemAdviseSetPreferredLocation,
					 cudaCpuDeviceId));
	} else if (mode == "uvm_advise_access") {
		// Declare that GPU will access this memory
		CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy,
					 dev));
	}
}

// ============================================================================
// Generic timing and statistics template
// ============================================================================

template <typename LaunchFunc>
inline void time_kernel(LaunchFunc launch_kernel, int warmup_iterations,
			int timed_iterations, std::vector<float> &runtimes,
			KernelResult &result)
{
	// Defensive check for invalid iterations
	if (timed_iterations <= 0) {
		result.median_ms = result.min_ms = result.max_ms = 0.0f;
		return;
	}

	// Warmup
	for (int i = 0; i < warmup_iterations; ++i) {
		launch_kernel();
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// Timed iterations
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	for (int i = 0; i < timed_iterations; ++i) {
		CUDA_CHECK(cudaEventRecord(start));
		launch_kernel();
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float ms;
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
		runtimes.push_back(ms);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	// Compute statistics
	std::sort(runtimes.begin(), runtimes.end());
	result.median_ms = runtimes[runtimes.size() / 2];
	result.min_ms = runtimes.front();
	result.max_ms = runtimes.back();
}

// ============================================================================
// Chunk-based kernel design parameters
// ============================================================================
// All kernels use a unified abstraction:
// - T: active threads (fixed)
// - chunk_elems: elements per chunk (aligned to pages)
// - chunks_per_thread: how many chunks each thread handles
// - stride_bytes: access granularity within chunk (4B or 4KB)
//
// Total working set: W ≈ T * chunks_per_thread * chunk_elems * sizeof(type)
// ============================================================================

// ============================================================================
// Kernel 1: Sequential - Chunk-sequential stream
// ============================================================================
// Goal: Model GEMM/stencil where each thread/warp sequentially scans its tile
// Each thread handles chunks_per_thread contiguous chunks, accessing them
// sequentially
__global__ void seq_chunk_kernel(const float *input, float *output, size_t N,
				 size_t chunk_elems, int chunks_per_thread,
				 size_t stride_elems, int prefetch_pages)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const size_t elems_per_page = 4096 / sizeof(float);

	// for (int c = 0; c < chunks_per_thread; ++c) {
	// 	size_t chunk_id = (size_t)tid * chunks_per_thread + c;
	// 	size_t chunk_start = chunk_id * chunk_elems;

	// 	if (chunk_start >= N)
	// 		break;

	// 	for (int p = 0; p < prefetch_pages; ++p) {
	// 		size_t pf_addr = chunk_start + p * elems_per_page;
	// 		if (pf_addr < N) {
	// 			prefetch_l2(&input[pf_addr]);
	// 			prefetch_l2(&output[pf_addr]);
	// 		}
	// 	}
	// }

	for (int c = 0; c < chunks_per_thread; ++c) {
		size_t chunk_id = (size_t)tid * chunks_per_thread + c;
		size_t chunk_start = chunk_id * chunk_elems;
		size_t chunk_end = min(chunk_start + chunk_elems, N);

		if (chunk_start >= N)
			break;

		for (size_t i = chunk_start; i < chunk_end; i += stride_elems) {
			if (i >= N)
				break;
			float val = input[i];
			val = val * 1.5f + 2.0f;
			output[i] = val;
		}
	}
}

struct RunSeqConfig {
	int numBlocks;
	int blockSize;
	float *input;
	float *output;
	size_t N;
	size_t chunk_elems;
	int chunks_per_thread;
	size_t stride_elems;
	int prefetch_pages;
};

extern "C" void launch_run_seq_kernel(RunSeqConfig *config)
{
	std::cout << "config->N=" << config->N
		  << ", config->chunk_elems=" << config->chunk_elems
		  << ", config->chunks_per_thread=" << config->chunks_per_thread
		  << ", config->stride_elems=" << config->stride_elems
		  << ", config->prefetch_pages=" << config->prefetch_pages
		  << std::endl;
	seq_chunk_kernel<<<config->numBlocks, config->blockSize>>>(
		config->input, config->output, config->N, config->chunk_elems,
		config->chunks_per_thread, config->stride_elems,
		config->prefetch_pages);
}

inline void run_seq_stream(size_t total_working_set, const std::string &mode,
			   size_t stride_bytes, int iterations,
			   std::vector<float> &runtimes, KernelResult &result)
{
	// Split: input (50%) + output (50%)
	size_t array_bytes = total_working_set / 2;
	size_t N = array_bytes / sizeof(float);
	size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

	// Sanity check for device mode
	if (mode == "device") {
		size_t free_bytes, total_bytes;
		CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
		if (total_working_set > free_bytes * 0.8) {
			throw std::runtime_error(
				"Working set too large for device mode");
		}
	}

	float *input, *output;

	// Allocate based on mode
	if (mode == "device") {
		CUDA_CHECK(cudaMalloc(&input, array_bytes));
		CUDA_CHECK(cudaMalloc(&output, array_bytes));
	} else {
		CUDA_CHECK(cudaMallocManaged(&input, array_bytes));
		CUDA_CHECK(cudaMallocManaged(&output, array_bytes));
	}

	// Initialize data
	if (mode == "device") {
		std::vector<float> host_data(N, 1.0f);
		CUDA_CHECK(cudaMemcpy(input, host_data.data(), array_bytes,
				      cudaMemcpyHostToDevice));
	} else {
		for (size_t i = 0; i < N; ++i) {
			input[i] = 1.0f;
		}
	}

	// Apply UVM hints (prefetch, advise, etc.)
	if (mode != "device" && mode != "uvm") {
		int dev;
		CUDA_CHECK(cudaGetDevice(&dev));
		apply_uvm_hints(input, array_bytes, mode, dev);
		apply_uvm_hints(output, array_bytes, mode, dev);
		if (mode == "uvm_prefetch") {
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

	// Chunk-based configuration
	// Fixed active threads: 8 * #SM * 32 (adjust based on GPU)
	int blockSize = 256;
	int numBlocks = 256; // Total ~64K threads
	int T = numBlocks * blockSize;
	int chunks_per_thread = 1;

	// Calculate chunk size (aligned to pages)
	size_t chunk_elems =
		(N + T * chunks_per_thread - 1) / (T * chunks_per_thread);
	size_t elems_per_page = 4096 / sizeof(float);
	chunk_elems = ((chunk_elems + elems_per_page - 1) / elems_per_page) *
		      elems_per_page;
	RunSeqConfig config{ .numBlocks = numBlocks,
			     .blockSize = blockSize,
			     .input = input,
			     .output = output,
			     .N = N,
			     .chunk_elems = chunk_elems,
			     .chunks_per_thread = chunks_per_thread,
			     .stride_elems = stride_elems,
			     .prefetch_pages = 4 };
	auto launch = [&]() { launch_run_seq_kernel(&config); };

	time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

	// Calculate bytes accessed
	size_t num_accesses = (N + stride_elems - 1) / stride_elems;
	if (stride_bytes >= 4096) {
		// Page-level: count UVM migration bytes
		size_t num_pages = num_accesses;
		result.bytes_accessed = num_pages * 4096 * 2; // input + output
	} else {
		// Element-level: count logical bytes
		result.bytes_accessed = num_accesses * sizeof(float) * 2;
	}

	// Cleanup
	CUDA_CHECK(cudaFree(input));
	CUDA_CHECK(cudaFree(output));
}

// ============================================================================
// Kernel 1b: Sequential with Device-side Prefetch (PTX prefetch.global.L2)
// ============================================================================
// Goal: Test if GPU-initiated prefetch can trigger UVM page migration early
// Uses PTX prefetch.global.L2 instruction to prefetch pages ahead of actual
// access This should trigger page faults for UVM pages on CPU, causing
// migration

__global__ void seq_prefetch_kernel(const float *input, float *output, size_t N,
				    size_t chunk_elems, int chunks_per_thread,
				    size_t stride_elems,
				    int prefetch_distance_pages)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const size_t elems_per_page = 4096 / sizeof(float); // 1024 floats per
							    // page
	const size_t prefetch_stride = 128 / sizeof(float); // 每128字节预取一次
							    // (cache line size)

	for (int c = 0; c < chunks_per_thread; ++c) {
		size_t chunk_id = (size_t)tid * chunks_per_thread + c;
		size_t chunk_start = chunk_id * chunk_elems;
		size_t chunk_end = min(chunk_start + chunk_elems, N);

		if (chunk_start >= N)
			continue;

		size_t chunk_size = chunk_end - chunk_start;
		size_t pages_in_chunk =
			(chunk_size + elems_per_page - 1) / elems_per_page;

		// Process page by page
		for (size_t page_idx = 0; page_idx < pages_in_chunk;
		     ++page_idx) {
// Step 1: Loop prefetch - 预取前 prefetch_distance_pages 页
// 按照 stride 在每一页内进行多点预取，覆盖整页
#pragma unroll 4
			for (int pf_offset = 1;
			     pf_offset <= prefetch_distance_pages;
			     ++pf_offset) {
				size_t prefetch_page = page_idx + pf_offset;
				if (prefetch_page < pages_in_chunk) {
					size_t prefetch_page_start =
						chunk_start +
						prefetch_page * elems_per_page;
					size_t prefetch_page_end =
						min(prefetch_page_start +
							    elems_per_page,
						    chunk_end);

					// 按照 stride 在这一页内循环预取多个点
					// 这样可以触发整页的预取，而不只是页的开头
					for (size_t pf_elem =
						     prefetch_page_start;
					     pf_elem < prefetch_page_end;
					     pf_elem += prefetch_stride) {
						if (pf_elem < N) {
							prefetch_l2(
								&input[pf_elem]);
							prefetch_l2(
								&output[pf_elem]);
						}
					}
				}
			}

			// Step 2: Process current page with stride
			size_t page_start =
				chunk_start + page_idx * elems_per_page;
			size_t page_end =
				min(page_start + elems_per_page, chunk_end);

			for (size_t i = page_start; i < page_end;
			     i += stride_elems) {
				if (i >= N)
					break;
				float val = input[i];
				val = val * 1.5f + 2.0f; // Light computation
				output[i] = val;
			}
		}
	}
}
inline void run_seq_device_prefetch(size_t total_working_set,
				    const std::string &mode,
				    size_t stride_bytes, int iterations,
				    std::vector<float> &runtimes,
				    KernelResult &result)
{
	// Split: input (50%) + output (50%)
	size_t array_bytes = total_working_set / 2;
	size_t N = array_bytes / sizeof(float);
	size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

	// This kernel is designed for UVM modes only
	if (mode == "device") {
		throw std::runtime_error(
			"seq_device_prefetch is designed for UVM modes, not device mode");
	}

	float *input, *output;

	// Always use managed memory
	CUDA_CHECK(cudaMallocManaged(&input, array_bytes));
	CUDA_CHECK(cudaMallocManaged(&output, array_bytes));

	// Initialize data on CPU (ensures pages start on CPU for UVM test)
	for (size_t i = 0; i < N; ++i) {
		input[i] = 1.0f;
	}

	// For this kernel, we specifically do NOT call cudaMemPrefetchAsync
	// We want pages to start on CPU and let the kernel's prefetch
	// instructions trigger the UVM migration

	// Chunk-based configuration
	int blockSize = 256;
	int numBlocks = 256;
	int T = numBlocks * blockSize;
	int chunks_per_thread = 1;

	// Calculate chunk size (aligned to pages)
	size_t chunk_elems =
		(N + T * chunks_per_thread - 1) / (T * chunks_per_thread);
	size_t elems_per_page = 4096 / sizeof(float);
	chunk_elems = ((chunk_elems + elems_per_page - 1) / elems_per_page) *
		      elems_per_page;

	// Prefetch distance: how many pages ahead to prefetch
	// Start with 4 pages ahead
	int prefetch_distance_pages = 4;

	auto launch = [&]() {
		seq_prefetch_kernel<<<numBlocks, blockSize>>>(
			input, output, N, chunk_elems, chunks_per_thread,
			stride_elems, prefetch_distance_pages);
	};

	time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

	// Calculate bytes accessed (same as seq_stream)
	size_t num_accesses = (N + stride_elems - 1) / stride_elems;
	if (stride_bytes >= 4096) {
		size_t num_pages = num_accesses;
		result.bytes_accessed = num_pages * 4096 * 2;
	} else {
		result.bytes_accessed = num_accesses * sizeof(float) * 2;
	}

	// Cleanup
	CUDA_CHECK(cudaFree(input));
	CUDA_CHECK(cudaFree(output));
}

// ============================================================================
// Kernel 2: Random - Per-thread random page tiles
// ============================================================================
// Goal: Model gather/scatter or GNN where each thread randomly accesses pages
// KEY: Randomness is at PAGE LEVEL, not element level

__device__ __forceinline__ unsigned int lcg_random(unsigned int x)
{
	return 1664525u * x + 1013904223u;
}

__global__ void rand_chunk_kernel(const float *input, float *output, size_t N,
				  size_t chunk_elems, int chunks_per_thread,
				  size_t stride_elems, unsigned int base_seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int T = gridDim.x * blockDim.x;

	unsigned int seed = base_seed ^ tid;
	size_t elems_per_page = 4096 / sizeof(float);

	for (int c = 0; c < chunks_per_thread; ++c) {
		size_t chunk_id = (size_t)tid * chunks_per_thread + c;
		size_t chunk_start = chunk_id * chunk_elems;
		size_t chunk_end = min(chunk_start + chunk_elems, N);

		if (chunk_start >= N)
			continue;

		// Calculate pages in this chunk
		size_t chunk_size = chunk_end - chunk_start;
		size_t pages_in_chunk =
			(chunk_size + elems_per_page - 1) / elems_per_page;

		// Access pages in RANDOM order (permutation-based, no
		// duplicates) Use multiplicative congruential permutation to
		// visit each page exactly once
		seed = lcg_random(seed);
		size_t step = (seed | 1u); // Ensure odd number (coprime with
					   // any power of 2)
		size_t offset = lcg_random(seed ^ 0xDEADBEEF) % pages_in_chunk;

		for (size_t p = 0; p < pages_in_chunk; ++p) {
			size_t random_page =
				(offset + p * step) % pages_in_chunk;
			size_t page_start =
				chunk_start + (random_page * elems_per_page);

			if (page_start >= N)
				continue;

			// Access one element in this page (page-level probing)
			size_t i = page_start;
			float val = input[i];
			val = val * 1.5f + 2.0f;
			output[i] = val;
		}
	}
}

inline void run_rand_stream(size_t total_working_set, const std::string &mode,
			    size_t stride_bytes, int iterations,
			    std::vector<float> &runtimes, KernelResult &result)
{
	// Split: input (50%) + output (50%)
	size_t array_bytes = total_working_set / 2;
	size_t N = array_bytes / sizeof(float);
	size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

	// Sanity check for device mode
	if (mode == "device") {
		size_t free_bytes, total_bytes;
		CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
		if (total_working_set > free_bytes * 0.8) {
			throw std::runtime_error(
				"Working set too large for device mode");
		}
	}

	float *input, *output;

	// Allocate based on mode
	if (mode == "device") {
		CUDA_CHECK(cudaMalloc(&input, array_bytes));
		CUDA_CHECK(cudaMalloc(&output, array_bytes));
	} else {
		CUDA_CHECK(cudaMallocManaged(&input, array_bytes));
		CUDA_CHECK(cudaMallocManaged(&output, array_bytes));
	}

	// Initialize data
	if (mode == "device") {
		std::vector<float> host_data(N, 1.0f);
		CUDA_CHECK(cudaMemcpy(input, host_data.data(), array_bytes,
				      cudaMemcpyHostToDevice));
	} else {
		for (size_t i = 0; i < N; ++i) {
			input[i] = 1.0f;
		}
	}

	// Apply UVM hints (prefetch, advise, etc.)
	if (mode != "device" && mode != "uvm") {
		int dev;
		CUDA_CHECK(cudaGetDevice(&dev));
		apply_uvm_hints(input, array_bytes, mode, dev);
		apply_uvm_hints(output, array_bytes, mode, dev);
		if (mode == "uvm_prefetch") {
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

	// Chunk-based configuration (same as sequential)
	int blockSize = 256;
	int numBlocks = 256;
	int T = numBlocks * blockSize;
	int chunks_per_thread = 1;

	size_t chunk_elems =
		(N + T * chunks_per_thread - 1) / (T * chunks_per_thread);
	size_t elems_per_page = 4096 / sizeof(float);
	chunk_elems = ((chunk_elems + elems_per_page - 1) / elems_per_page) *
		      elems_per_page;

	unsigned int base_seed = 0x12345678;

	auto launch = [&]() {
		rand_chunk_kernel<<<numBlocks, blockSize>>>(
			input, output, N, chunk_elems, chunks_per_thread,
			stride_elems, base_seed);
	};

	time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

	// Calculate bytes accessed
	size_t total_pages = (N * sizeof(float) + 4095) / 4096;
	if (stride_bytes >= 4096) {
		// Page-level: count UVM migration bytes
		result.bytes_accessed = total_pages * 4096 * 2; // input +
								// output
	} else {
		// Element-level
		result.bytes_accessed = N * sizeof(float) * 2;
	}

	// Cleanup
	CUDA_CHECK(cudaFree(input));
	CUDA_CHECK(cudaFree(output));
}

// ============================================================================
// Kernel 3: Pointer Chase - Chunk-local pointer chains
// ============================================================================
// Goal: Model pointer-heavy workloads (graph traversal, linked list)
// Each chunk has its own independent random permutation

struct Node {
	unsigned int next; // Index of next node
	float data;
	float padding[1]; // Align to 16 bytes
};

// GPU-based chunk initialization: each chunk is a local random permutation
__global__ void init_chunks_kernel(Node *nodes, size_t nodes_per_chunk,
				   int total_chunks)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = gridDim.x * blockDim.x;
	size_t total_nodes = (size_t)total_chunks * nodes_per_chunk;

	for (size_t i = tid; i < total_nodes; i += stride) {
		size_t chunk_id = i / nodes_per_chunk;
		size_t offset = i % nodes_per_chunk;

		// Generate next pointer within the same chunk (chunk-local
		// chain)
		unsigned int r = lcg_random((unsigned int)i);
		size_t next_offset = r % nodes_per_chunk;
		size_t next_idx = chunk_id * nodes_per_chunk + next_offset;

		nodes[i].next = (unsigned int)next_idx;
		nodes[i].data = 1.0f;
		nodes[i].padding[0] = 0.0f;
	}
}

__global__ void pointer_chunk_kernel(const Node *nodes, float *output,
				     size_t nodes_per_chunk,
				     int chunks_per_thread, int chase_steps)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int T = gridDim.x * blockDim.x;

	// Each thread processes chunks_per_thread chunks
	for (int c = 0; c < chunks_per_thread; ++c) {
		size_t chunk_id = (size_t)tid * chunks_per_thread + c;
		size_t chunk_start = chunk_id * nodes_per_chunk;

		// Start from beginning of chunk
		unsigned int cur = (unsigned int)chunk_start;
		float sum = 0.0f;

// Chase pointers (dependent loads)
#pragma unroll 4
		for (int s = 0; s < chase_steps; ++s) {
			sum += nodes[cur].data;
			cur = nodes[cur].next;
		}

		output[chunk_id] = sum;
	}
}

inline void run_pointer_chase(size_t total_working_set, const std::string &mode,
			      size_t stride_bytes, int iterations,
			      std::vector<float> &runtimes,
			      KernelResult &result)
{
	// Split: nodes (90%) + output (10%)
	size_t nodes_bytes = static_cast<size_t>(total_working_set * 0.9);
	size_t output_bytes = static_cast<size_t>(total_working_set * 0.1);

	size_t total_nodes = nodes_bytes / sizeof(Node);

	(void)stride_bytes; // Pointer chase ignores stride - always follows
			    // pointers

	// Sanity check for device mode
	if (mode == "device") {
		size_t free_bytes, total_bytes;
		CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
		if (nodes_bytes + output_bytes > free_bytes * 0.8) {
			throw std::runtime_error(
				"Working set too large for device mode");
		}
	}

	// Chunk configuration
	int blockSize = 256;
	int numBlocks = 256;
	int T = numBlocks * blockSize;
	int chunks_per_thread = 1;
	int total_chunks = T * chunks_per_thread;

	// Calculate nodes per chunk
	size_t nodes_per_chunk =
		(total_nodes + total_chunks - 1) / total_chunks;
	// Align to page boundary
	size_t nodes_per_page = 4096 / sizeof(Node);
	nodes_per_chunk =
		((nodes_per_chunk + nodes_per_page - 1) / nodes_per_page) *
		nodes_per_page;

	// Actual allocation
	size_t n_alloc = (size_t)total_chunks * nodes_per_chunk;
	nodes_bytes = n_alloc * sizeof(Node);
	output_bytes = total_chunks * sizeof(float);

	Node *nodes;
	float *output_array;

	// Allocate
	if (mode == "device") {
		CUDA_CHECK(cudaMalloc(&nodes, nodes_bytes));
		CUDA_CHECK(cudaMalloc(&output_array, output_bytes));
	} else {
		CUDA_CHECK(cudaMallocManaged(&nodes, nodes_bytes));
		CUDA_CHECK(cudaMallocManaged(&output_array, output_bytes));
	}

	// Initialize chunks on GPU
	int init_blockSize = 256;
	int init_numBlocks = (n_alloc + init_blockSize - 1) / init_blockSize;
	init_numBlocks = std::min(init_numBlocks, 2048);

	init_chunks_kernel<<<init_numBlocks, init_blockSize>>>(
		nodes, nodes_per_chunk, total_chunks);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Apply UVM hints (prefetch, advise, etc.)
	if (mode != "device" && mode != "uvm") {
		int dev;
		CUDA_CHECK(cudaGetDevice(&dev));
		apply_uvm_hints(nodes, nodes_bytes, mode, dev);
		apply_uvm_hints(output_array, output_bytes, mode, dev);
		if (mode == "uvm_prefetch") {
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

	// Pointer chase steps: keep it small for reasonable runtime
	int chase_steps = 8;

	auto launch = [&]() {
		pointer_chunk_kernel<<<numBlocks, blockSize>>>(
			nodes, output_array, nodes_per_chunk, chunks_per_thread,
			chase_steps);
	};

	time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

	// Calculate bytes accessed based on actual logical accesses
	// Each thread accesses chase_steps nodes
	size_t logical_accesses =
		(size_t)total_chunks * chunks_per_thread * chase_steps;
	size_t logical_bytes = logical_accesses * sizeof(Node);

	// Estimate pages touched (logical bytes / page size)
	size_t pages_touched = (logical_bytes + 4095) / 4096;

	// Upper bound: cannot exceed total pages in nodes array
	size_t total_pages = (n_alloc * sizeof(Node) + 4095) / 4096;
	pages_touched = std::min(pages_touched, total_pages);

	// For UVM, migration granularity is page-level
	result.bytes_accessed = pages_touched * 4096;

	// Cleanup
	CUDA_CHECK(cudaFree(nodes));
	CUDA_CHECK(cudaFree(output_array));
}

#endif // SYNTHETIC_CUH
