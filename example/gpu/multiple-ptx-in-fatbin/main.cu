#include <cuda_runtime.h>
#include <stdio.h>

extern __global__ void vectorAdd(float *a, float *b, float *c, int n);
extern __global__ void vectorMul(float *a, float *b, float *c, int n);
extern __global__ void vectorSum(float *input, float *output, int n);

extern __device__ int shared_counter;
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float h_sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    float *d_a, *d_b, *d_c, *d_sum;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice));
    
    int counter_init = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(shared_counter, &counter_init, sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    
    // 启动 Kernel 1: 向量加法
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_c[i]);
    }
    puts("\n\n");    
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_c[i]);
    }
    printf("\n\n");
    
    int smemSize = threadsPerBlock * sizeof(float);
    vectorSum<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_a, d_sum, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sum: %.1f\n\n", h_sum);
    
    int final_counter;
    CHECK_CUDA(cudaMemcpyFromSymbol(&final_counter, shared_counter, sizeof(int)));
    printf("Shared counter: %d\n", final_counter);
    printf("(should be equal to N + N + blocksPerGrid = %d)\n", N + N + blocksPerGrid);
    
    // 清理
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_sum);
    
    return 0;
}
