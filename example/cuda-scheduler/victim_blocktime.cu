// 文件：bfs_cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <random>

// Block allocation configuration
struct BlockAllocator {
    static constexpr int NUM_TENANTS = 3;  // Number of tenants
    static constexpr int TOTAL_BLOCKS = 1024;  // Total number of blocks available
    static constexpr int BLOCKS_PER_TENANT = TOTAL_BLOCKS / NUM_TENANTS;  // Equal division
    
    static int get_tenant_blocks(int tenant_id) {
        return BLOCKS_PER_TENANT;
    }
};

// CUDA kernel: 扫描当前 frontier，松弛所有邻居，构建 next frontier
__global__
void bfs_kernel(const int *row_ptr,
                const int *col_ind,
                const int  *frontier,
                int          frontier_size,
                int        *dist,
                int        *next_frontier,
                int        *next_frontier_size,
                int          current_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int u = frontier[idx];
    int start = row_ptr[u];
    int end   = row_ptr[u+1];
    for (int ei = start; ei < end; ++ei) {
        int v = col_ind[ei];
        // 原子判断并设置距离
        if (atomicCAS(&dist[v], -1, current_dist+1) == -1) {
            // 只有当 dist[v] 原先为 -1 时，才算新加入 frontier
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = v;
        }
    }
}

void bfs_cuda(int n,
              const std::vector<int>& row_ptr_h,
              const std::vector<int>& col_ind_h,
              int source,
              std::vector<int>& dist_h,
              int tenant_id)  // Added tenant_id parameter
{
    // 1) 申请并拷贝图到设备
    int *d_row_ptr, *d_col_ind;
    cudaMalloc(&d_row_ptr, (n+1)*sizeof(int));
    cudaMalloc(&d_col_ind, col_ind_h.size()*sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr_h.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind_h.data(), col_ind_h.size()*sizeof(int), cudaMemcpyHostToDevice);

    // 2) 申请并初始化 distance 数组
    int *d_dist;
    cudaMalloc(&d_dist, n*sizeof(int));
    cudaMemset(d_dist, -1, n*sizeof(int));

    // 3) 申请 frontier 和 next_frontier 空间
    int *d_frontier, *d_next_frontier;
    cudaMalloc(&d_frontier, n*sizeof(int));
    cudaMalloc(&d_next_frontier, n*sizeof(int));

    // 4) 申请并初始化 frontier 大小变量
    int *d_frontier_size, *d_next_frontier_size;
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    // 5) 初始化第 0 轮 frontier
    int h_frontier_size = 1;
    cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);
    // source 的距离置为 0
    cudaMemcpy(&d_dist[source], &zero, sizeof(int), cudaMemcpyHostToDevice);

    int current_dist = 0;
    // 6) 迭代，多轮 kernel launch
    while (true) {
        // 读出当前 frontier_size
        cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_frontier_size == 0) break;

        int threads = 256;
        // Use proportional block allocation
        int blocks = std::min(BlockAllocator::get_tenant_blocks(tenant_id),
                            (h_frontier_size + threads - 1) / threads);
        
        // 每轮开始前，清 next_frontier_size
        cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // 运行 BFS kernel with allocated blocks
        bfs_kernel<<<blocks, threads>>>(
            d_row_ptr, d_col_ind,
            d_frontier, h_frontier_size,
            d_dist,
            d_next_frontier, d_next_frontier_size,
            current_dist
        );
        cudaDeviceSynchronize();

        // 交换 frontier 指针与大小
        std::swap(d_frontier, d_next_frontier);
        cudaMemcpy(d_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        // 下一轮距离递增
        ++current_dist;
    }

    // 7) 将结果拷回 host
    dist_h.resize(n);
    cudaMemcpy(dist_h.data(), d_dist, n*sizeof(int), cudaMemcpyDeviceToHost);

    // 8) 释放资源
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier_size);
}

// Global variables for synchronization and timing
std::mutex cout_mutex;
std::atomic<long long> total_latency(0);
std::atomic<int> completed_threads(0);

// 生成随机图的函数
void generate_random_graph(int n, int avg_degree, std::vector<int>& row_ptr, std::vector<int>& col_ind) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n-1);
    
    // 初始化row_ptr
    row_ptr.resize(n + 1, 0);
    
    // 为每个节点生成边
    std::vector<std::vector<int>> edges(n);
    for (int i = 0; i < n; i++) {
        // 每个节点平均连接到avg_degree个其他节点
        int degree = avg_degree;
        for (int j = 0; j < degree; j++) {
            int target = dis(gen);
            if (target != i) {  // 避免自环
                edges[i].push_back(target);
            }
        }
    }
    
    // 构建CSR格式
    int edge_count = 0;
    for (int i = 0; i < n; i++) {
        row_ptr[i] = edge_count;
        edge_count += edges[i].size();
    }
    row_ptr[n] = edge_count;
    
    // 构建col_ind
    col_ind.resize(edge_count);
    int pos = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < edges[i].size(); j++) {
            col_ind[pos++] = edges[i][j];
        }
    }
}

void thread_function(int thread_id) {
    const int n = 10'000'000;  // 10M nodes
    const int avg_degree = 10;  // 平均每个节点连接到10个其他节点
    
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    
    // 生成随机图
    generate_random_graph(n, avg_degree, row_ptr, col_ind);
    
    int source = 0;
    std::vector<int> dist;
    
    long long thread_total_latency = 0;
    
    for(int i = 0; i < 200; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        bfs_cuda(n, row_ptr, col_ind, source, dist, thread_id);  // Pass tenant_id
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        thread_total_latency += duration.count();
        
        if(i % 20 == 0) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread " << thread_id << " completed iteration " << i 
                      << " (Graph size: " << n << " nodes, " 
                      << row_ptr[n] << " edges, Blocks: " 
                      << BlockAllocator::get_tenant_blocks(thread_id) << ")" << std::endl;
        }
    }
    
    total_latency += thread_total_latency;
    completed_threads++;
    
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "Thread " << thread_id << " completed. Average latency: " 
              << (thread_total_latency / 200.0) << " microseconds" << std::endl;
}

int main() {
    std::vector<std::thread> threads;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create 3 threads
    for(int i = 0; i < 3; i++) {
        threads.emplace_back(thread_function, i);
    }
    
    // Wait for all threads to complete
    for(auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "Average latency per kernel launch: " << (total_latency / (3.0 * 200)) << " microseconds" << std::endl;
    
    return 0;
}