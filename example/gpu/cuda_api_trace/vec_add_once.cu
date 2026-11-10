#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    constexpr int N = 1 << 12;
    constexpr size_t bytes = N * sizeof(float);
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(3 * i);
    }

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    if (cudaMalloc(&dA, bytes) != cudaSuccess ||
        cudaMalloc(&dB, bytes) != cudaSuccess ||
        cudaMalloc(&dC, bytes) != cudaSuccess) {
        return 2;
    }
    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);
    vectorAdd<<<N / 128, 128>>>(dA, dB, dC, N);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 3;
    }
    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (C[i] != static_cast<float>(4 * i)) {
            fprintf(stderr, "mismatch at %d: %f\n", i, C[i]);
            return 4;
        }
    }
    printf("vectorAdd passed: C[0]=%.1f C[1]=%.1f\n", C[0], C[1]);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
