#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
	do {                                                                   \
		cudaError_t err = call;                                        \
		if (err != cudaSuccess) {                                      \
			throw std::runtime_error(std::string("CUDA error: ") + \
						 cudaGetErrorString(err) +     \
						 " at " + __FILE__ + ":" +     \
						 std::to_string(__LINE__));    \
		}                                                              \
	} while (0)

// Result structure for kernel benchmarking
struct KernelResult {
	float min_time_ms;
	float max_time_ms;
	float avg_time_ms;
	float median_time_ms;
	size_t bytes_accessed;

	float get_bandwidth_gbps() const
	{
		return (bytes_accessed / 1e9) / (avg_time_ms / 1e3);
	}
};

// Timing utility function
template <typename LaunchFunc>
void time_kernel(LaunchFunc launch, int warmup, int iterations,
		 std::vector<float> &runtimes, KernelResult &result)
{
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// Warmup runs
	for (int i = 0; i < warmup; ++i) {
		launch();
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	runtimes.clear();
	runtimes.reserve(iterations);

	// Timed runs
	for (int i = 0; i < iterations; ++i) {
		CUDA_CHECK(cudaEventRecord(start));
		launch();
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
		runtimes.push_back(ms);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	// Calculate statistics
	std::vector<float> sorted_times = runtimes;
	std::sort(sorted_times.begin(), sorted_times.end());

	result.min_time_ms = sorted_times.front();
	result.max_time_ms = sorted_times.back();

	float sum = 0.0f;
	for (float t : runtimes)
		sum += t;
	result.avg_time_ms = sum / runtimes.size();

	size_t mid = sorted_times.size() / 2;
	if (sorted_times.size() % 2 == 0) {
		result.median_time_ms =
			(sorted_times[mid - 1] + sorted_times[mid]) / 2.0f;
	} else {
		result.median_time_ms = sorted_times[mid];
	}
}

// Device prefetch instruction
__device__ __forceinline__ void prefetch_l2(const void *addr)
{
	asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
}

__global__ void seq_prefetch_kernel(const float *input, float *output, size_t N,
				    size_t chunk_elems, int chunks_per_thread,
				    size_t stride_elems,
				    int prefetch_distance_pages)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t elems_per_page = 4096 / sizeof(float);

	// 每次处理的页面批次大小
	const int BATCH_SIZE = 4;
	// 预取窗口大小（批次数）
	const int PREFETCH_BATCHES = 2;

	for (int c = 0; c < chunks_per_thread; ++c) {
		size_t chunk_id = (size_t)tid * chunks_per_thread + c;
		size_t chunk_start = chunk_id * chunk_elems;
		size_t chunk_end = min(chunk_start + chunk_elems, N);

		if (chunk_start >= N)
			continue;

		size_t chunk_size = chunk_end - chunk_start;
		size_t pages_in_chunk =
			(chunk_size + elems_per_page - 1) / elems_per_page;
		size_t batches_in_chunk =
			(pages_in_chunk + BATCH_SIZE - 1) / BATCH_SIZE;

		for (int b = 0;
		     b < PREFETCH_BATCHES && b < (int)batches_in_chunk; ++b) {
#pragma unroll
			for (int p = 0; p < BATCH_SIZE; ++p) {
				size_t page = b * BATCH_SIZE + p;
				if (page < pages_in_chunk) {
					size_t prefetch_elem =
						chunk_start +
						page * elems_per_page;
					if (prefetch_elem < N) {
						prefetch_l2(
							&input[prefetch_elem]);
						prefetch_l2(
							&output[prefetch_elem]);
					}
				}
			}
		}

		// 按批次处理
		for (size_t batch_idx = 0; batch_idx < batches_in_chunk;
		     ++batch_idx) {
			// 预取未来批次
			size_t prefetch_batch = batch_idx + PREFETCH_BATCHES;
			if (prefetch_batch < batches_in_chunk) {
#pragma unroll
				for (int p = 0; p < BATCH_SIZE; ++p) {
					size_t page =
						prefetch_batch * BATCH_SIZE + p;
					if (page < pages_in_chunk) {
						size_t prefetch_elem =
							chunk_start +
							page * elems_per_page;
						if (prefetch_elem < N) {
							prefetch_l2(
								&input[prefetch_elem]);
							prefetch_l2(
								&output[prefetch_elem]);
						}
					}
				}
			}

// 处理当前批次的所有页面
#pragma unroll
			for (int p = 0; p < BATCH_SIZE; ++p) {
				size_t page_idx = batch_idx * BATCH_SIZE + p;
				if (page_idx >= pages_in_chunk)
					break;

				size_t page_start =
					chunk_start + page_idx * elems_per_page;
				size_t page_end = min(
					page_start + elems_per_page, chunk_end);

				for (size_t i = page_start; i < page_end;
				     i += stride_elems) {
					if (i >= N)
						break;
					float val = input[i];
					val = val * 1.5f + 2.0f;
					output[i] = val;
				}
			}
		}
	}
}
// Run the sequential device prefetch benchmark
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

// Verify the results are correct
bool verify_results(size_t total_working_set)
{
	size_t array_bytes = total_working_set / 2;
	size_t N = array_bytes / sizeof(float);

	float *input, *output;
	CUDA_CHECK(cudaMallocManaged(&input, array_bytes));
	CUDA_CHECK(cudaMallocManaged(&output, array_bytes));

	for (size_t i = 0; i < N; ++i) {
		input[i] = 1.0f;
		output[i] = 0.0f;
	}

	int blockSize = 256;
	int numBlocks = 256;
	int T = numBlocks * blockSize;
	int chunks_per_thread = 1;
	size_t chunk_elems =
		(N + T * chunks_per_thread - 1) / (T * chunks_per_thread);
	size_t elems_per_page = 4096 / sizeof(float);
	chunk_elems = ((chunk_elems + elems_per_page - 1) / elems_per_page) *
		      elems_per_page;
	size_t stride_elems = 1;
	int prefetch_distance_pages = 4;

	seq_prefetch_kernel<<<numBlocks, blockSize>>>(
		input, output, N, chunk_elems, chunks_per_thread, stride_elems,
		prefetch_distance_pages);
	CUDA_CHECK(cudaDeviceSynchronize());

	bool success = true;
	float expected = 1.0f * 1.5f + 2.0f; // 3.5f
	size_t errors = 0;
	for (size_t i = 0; i < N && errors < 10; ++i) {
		if (std::abs(output[i] - expected) > 1e-5f) {
			std::cerr << "Verification failed at index " << i
				  << ": expected " << expected << ", got "
				  << output[i] << std::endl;
			success = false;
			errors++;
		}
	}

	CUDA_CHECK(cudaFree(input));
	CUDA_CHECK(cudaFree(output));

	return success;
}

void print_device_info()
{
	int device;
	CUDA_CHECK(cudaGetDevice(&device));

	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

	std::cout << "============================================"
		  << std::endl;
	std::cout << "Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor
		  << std::endl;
	std::cout << "Total Global Memory: "
		  << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB"
		  << std::endl;
	std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits"
		  << std::endl;
	std::cout << "L2 Cache Size: " << (prop.l2CacheSize / 1024) << " KB"
		  << std::endl;
	std::cout << "Unified Addressing: "
		  << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
	std::cout << "Managed Memory: " << (prop.managedMemory ? "Yes" : "No")
		  << std::endl;
	std::cout << "Concurrent Managed Access: "
		  << (prop.concurrentManagedAccess ? "Yes" : "No") << std::endl;
	std::cout << "============================================"
		  << std::endl;
}

int main(int argc, char *argv[])
{
	try {
		print_device_info();

		// Default parameters
		size_t total_working_set_mb = 256; // 256 MB total working set
		size_t stride_bytes = 4; // Access every float (no stride)
		int iterations = 10;
		std::string mode = "uvm"; // UVM mode (managed memory)

		// Parse command line arguments
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "--size" && i + 1 < argc) {
				total_working_set_mb = std::stoull(argv[++i]);
			} else if (arg == "--stride" && i + 1 < argc) {
				stride_bytes = std::stoull(argv[++i]);
			} else if (arg == "--iterations" && i + 1 < argc) {
				iterations = std::stoi(argv[++i]);
			} else if (arg == "--mode" && i + 1 < argc) {
				mode = argv[++i];
			} else if (arg == "--help") {
				std::cout << "Usage: " << argv[0]
					  << " [options]" << std::endl;
				std::cout << "Options:" << std::endl;
				std::cout
					<< "  --size <MB>        Total working set size in MB (default: 256)"
					<< std::endl;
				std::cout
					<< "  --stride <bytes>   Stride in bytes (default: 4)"
					<< std::endl;
				std::cout
					<< "  --iterations <N>   Number of timed iterations (default: 10)"
					<< std::endl;
				std::cout
					<< "  --mode <mode>      Memory mode: uvm (default: uvm)"
					<< std::endl;
				return 0;
			}
		}

		size_t total_working_set = total_working_set_mb * 1024 * 1024;

		std::cout << "\nBenchmark Configuration:" << std::endl;
		std::cout << "  Working Set Size: " << total_working_set_mb
			  << " MB" << std::endl;
		std::cout << "  Stride: " << stride_bytes << " bytes"
			  << std::endl;
		std::cout << "  Iterations: " << iterations << std::endl;
		std::cout << "  Mode: " << mode << std::endl;
		std::cout << std::endl;

		// Verify correctness first
		std::cout << "Verifying correctness..." << std::endl;
		if (verify_results(total_working_set)) {
			std::cout << "Verification PASSED" << std::endl;
		} else {
			std::cout << "Verification FAILED" << std::endl;
			return 1;
		}
		std::cout << std::endl;

		// Run benchmark
		std::vector<float> runtimes;
		KernelResult result;

		std::cout << "Running benchmark..." << std::endl;
		run_seq_device_prefetch(total_working_set, mode, stride_bytes,
					iterations, runtimes, result);

		// Print results
		std::cout << "\n============================================"
			  << std::endl;
		std::cout << "Results:" << std::endl;
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "  Min Time:     " << result.min_time_ms << " ms"
			  << std::endl;
		std::cout << "  Max Time:     " << result.max_time_ms << " ms"
			  << std::endl;
		std::cout << "  Avg Time:     " << result.avg_time_ms << " ms"
			  << std::endl;
		std::cout << "  Median Time:  " << result.median_time_ms
			  << " ms" << std::endl;
		std::cout << "  Bytes Accessed: "
			  << (result.bytes_accessed / (1024.0 * 1024.0))
			  << " MB" << std::endl;
		std::cout << "  Bandwidth:    " << result.get_bandwidth_gbps()
			  << " GB/s" << std::endl;
		std::cout << "============================================"
			  << std::endl;

		// Print individual run times
		std::cout << "\nIndividual Run Times (ms):" << std::endl;
		for (size_t i = 0; i < runtimes.size(); ++i) {
			std::cout << "  Run " << std::setw(2) << (i + 1) << ": "
				  << runtimes[i] << std::endl;
		}

	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
