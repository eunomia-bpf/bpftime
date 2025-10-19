// kernel2.cu
#include <cuda_runtime.h>

extern __device__ int shared_counter;

__global__ void vectorMul(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
        atomicAdd(&shared_counter, 1);
    }
}
