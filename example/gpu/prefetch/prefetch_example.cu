__device__ __forceinline__
void prefetch_l2(const void* addr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(addr));
}

__global__ void seq_prefetch_kernel(const float* input, float* output,
                                    size_t N,
                                    size_t chunk_elems,
                                    int chunks_per_thread,
                                    size_t stride_elems,
                                    int prefetch_distance_pages) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t elems_per_page = 4096 / sizeof(float);  // 1024 floats per page

    for (int c = 0; c < chunks_per_thread; ++c) {
        size_t chunk_id = (size_t)tid * chunks_per_thread + c;
        size_t chunk_start = chunk_id * chunk_elems;
        size_t chunk_end = min(chunk_start + chunk_elems, N);

        if (chunk_start >= N) continue;

        size_t chunk_size = chunk_end - chunk_start;
        size_t pages_in_chunk = (chunk_size + elems_per_page - 1) / elems_per_page;

        // Process page by page
        for (size_t page_idx = 0; page_idx < pages_in_chunk; ++page_idx) {
            // Step 1: Prefetch pages ahead (triggers UVM page fault if page is on CPU)
            size_t prefetch_page = page_idx + prefetch_distance_pages;
            if (prefetch_page < pages_in_chunk) {
                size_t prefetch_elem = chunk_start + prefetch_page * elems_per_page;
                if (prefetch_elem < N) {
                    prefetch_l2(&input[prefetch_elem]);
                    prefetch_l2(&output[prefetch_elem]);
                }
            }

            // Step 2: Process current page with stride
            size_t page_start = chunk_start + page_idx * elems_per_page;
            size_t page_end = min(page_start + elems_per_page, chunk_end);

            for (size_t i = page_start; i < page_end; i += stride_elems) {
                if (i >= N) break;
                float val = input[i];
                val = val * 1.5f + 2.0f;  // Light computation
                output[i] = val;
            }
        }
    }
}

inline void run_seq_device_prefetch(size_t total_working_set, const std::string& mode, size_t stride_bytes,
                                    int iterations, std::vector<float>& runtimes, KernelResult& result) {
    // Split: input (50%) + output (50%)
    size_t array_bytes = total_working_set / 2;
    size_t N = array_bytes / sizeof(float);
    size_t stride_elems = std::max(1UL, stride_bytes / sizeof(float));

    // This kernel is designed for UVM modes only
    if (mode == "device") {
        throw std::runtime_error("seq_device_prefetch is designed for UVM modes, not device mode");
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
    // We want pages to start on CPU and let the kernel's prefetch instructions
    // trigger the UVM migration

    // Chunk-based configuration
    int blockSize = 256;
    int numBlocks = 256;
    int T = numBlocks * blockSize;
    int chunks_per_thread = 1;

    // Calculate chunk size (aligned to pages)
    size_t chunk_elems = (N + T * chunks_per_thread - 1) / (T * chunks_per_thread);
    size_t elems_per_page = 4096 / sizeof(float);
    chunk_elems = ((chunk_elems + elems_per_page - 1) / elems_per_page) * elems_per_page;

    // Prefetch distance: how many pages ahead to prefetch
    // Start with 4 pages ahead
    int prefetch_distance_pages = 4;

    auto launch = [&]() {
        seq_prefetch_kernel<<<numBlocks, blockSize>>>(input, output, N, chunk_elems,
                                                       chunks_per_thread, stride_elems,
                                                       prefetch_distance_pages);
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
