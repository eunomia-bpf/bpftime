// kernel1.cu
#include <cuda_runtime.h>

__device__ int shared_counter = 0;

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        atomicAdd(&shared_counter, 1);
    }
}
