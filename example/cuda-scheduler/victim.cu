// 文件：bfs_cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>

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
              std::vector<int>& dist_h)
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
        int blocks  = (h_frontier_size + threads - 1) / threads;
        // 每轮开始前，清 next_frontier_size
        cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // 运行 BFS kernel
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

int main()
{
    // 示例：构造一个简单无向图
    // 0—1—2
    // |  |
    // 3—4
    int n = 5;
    std::vector<int> row_ptr = {0, 2, 5, 7, 9, 10};
    std::vector<int> col_ind = {
        1,3,    // 0
        0,2,4,  // 1
        1,4,    // 2
        0,4,    // 3
        1,2,3   // 4
    };

    int source = 0;
    std::vector<int> dist;
    bfs_cuda(n, row_ptr, col_ind, source, dist);

    std::cout << "Distances from node " << source << ":\n";
    for (int i = 0; i < n; ++i)
        std::cout << "  to " << i << " = " << dist[i] << "\n";
    return 0;
}