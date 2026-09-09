#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        if (auto error = (call); error != cudaSuccess) {                       \
            std::cerr << #call << " failed: " << cudaGetErrorString(error)     \
                      << '\n';                                                 \
            return 1;                                                          \
        }                                                                      \
    } while (false)

/*
nvcc -x cu -cuda vectorAdd.cu -o vectorAdd.cpp
python filter_hashtag.py
g++ vectorAdd-new.cpp -Wall -L /usr/local/cuda-12.6/lib64 -lcudart -o vectorAdd -g
 */

constexpr int BLOCK_SIZE = 10;

__constant__ int d_N;

// A simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const bool expect_patched = getenv("BPFTIME_EXPECT_PATCHED") != nullptr;
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
        return 1;

    const int h_N = BLOCK_SIZE;
    size_t bytes = h_N * sizeof(float);
    
    // Allocate and initialize host memory using vectors
    std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);
    
    for (int i = 0; i < h_N; ++i)
    {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    std::vector<float *> d_A(device_count), d_B(device_count), d_C(device_count);
    for (int device = 0; device < device_count; device++) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&d_A[device], bytes));
        CUDA_CHECK(cudaMalloc(&d_B[device], bytes));
        CUDA_CHECK(cudaMalloc(&d_C[device], bytes));
        CUDA_CHECK(cudaMemcpy(d_A[device], h_A.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[device], h_B.data(), bytes,
                              cudaMemcpyHostToDevice));
    }

    // Run the kernel in an infinite loop
    while (true) {
        for (int device = 0; device < device_count; device++) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N)));
            CUDA_CHECK(cudaMemset(d_C[device], 0, bytes));
            vectorAdd<<<BLOCK_SIZE, 1>>>(d_A[device], d_B[device], d_C[device]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C[device], bytes,
                                  cudaMemcpyDeviceToHost));
            std::cout << "GPU " << device << ": C[1] = " << h_C[1]
                      << ", C[7] = " << h_C[7]
                      << " (expected 0 or 3, 21)" << std::endl;
            if ((expect_patched ? h_C[1] != 0
                                : (h_C[1] != 0 && h_C[1] != 3)) ||
                h_C[7] != 21)
                return 1;
        }
        std::cout << "Validated all visible GPUs" << std::endl;
        // Sleep for 1 second
        sleep(1);
    }

}
