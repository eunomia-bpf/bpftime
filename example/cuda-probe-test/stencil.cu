#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 半径
constexpr int RADIUS = 1;
// stencil 权重，存放在常量内存中以便各线程快速访问
__constant__ float d_weights[2 * RADIUS + 1];

// GPU kernel：每个线程处理输出 y[i]
__global__ void stencil1D(const float* __restrict__ x,
                          float* __restrict__ y,
                          int N)
{
    extern __shared__ float s_data[];  
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 全局索引到共享内存索引的偏移（保留左右两个“halo”格）
    int s_idx = tid + RADIUS;

    // 1) 将主数据加载到 shared memory
    if (gid < N) {
        s_data[s_idx] = x[gid];
        // 左 halo
        if (tid < RADIUS) {
            int left_idx = max(gid - RADIUS, 0);
            s_data[tid] = x[left_idx];
        }
        // 右 halo
        if (tid >= blockDim.x - RADIUS) {
            int right_idx = min(gid + RADIUS, N - 1);
            s_data[s_idx + RADIUS] = x[right_idx];
        }
    }
    __syncthreads();

    // 2) 计算 stencil
    if (gid < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
            acc += d_weights[offset + RADIUS] * 
                   s_data[s_idx + offset];
        }
        y[gid] = acc;
    }
}

// Host 代码
int main()
{
    const int N = 1 << 20;  // 1M 元素
    const int bytes = N * sizeof(float);

    // 准备 host 数据
    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i % 100) / 100.0f;
    }
    // stencil 权重
    float h_weights[2 * RADIUS + 1] = {0.25f, 0.5f, 0.25f};

    // 分配 device 内存
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // 复制数据到 device
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_weights, h_weights, sizeof(h_weights));

    // 设置 kernel 参数
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    // shared memory 大小 = (blockSize + 2*RADIUS) * sizeof(float)
    size_t sharedBytes = (blockSize + 2 * RADIUS) * sizeof(float);

    // 调用 kernel
    stencil1D<<<gridSize, blockSize, sharedBytes>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // 拷回结果
    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    // 简单验证
    std::cout << "y[0] = "  << h_y[0]
              << ", y[N/2] = " << h_y[N/2]
              << ", y[N-1] = "  << h_y[N-1]
              << std::endl;

    // 释放
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}