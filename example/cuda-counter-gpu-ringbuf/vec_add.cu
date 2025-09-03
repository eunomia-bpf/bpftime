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

/*
nvcc -x cu -cuda vectorAdd.cu -o vectorAdd.cpp
python filter_hashtag.py
g++ vectorAdd-new.cpp -Wall -L /usr/local/cuda-12.6/lib64 -lcudart -o vectorAdd -g
 */

__constant__ int d_N;

// A simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle cases where total threads < N
    while (idx < d_N)
    {
        C[idx] = A[idx] + B[idx];
        idx += blockDim.x * gridDim.x;  // Grid stride
    }
}

int main()
{
    // Set vector size in constant memory
    const int h_N = 1 << 20;  // 1M elements
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));
    
    size_t bytes = h_N * sizeof(float);
    
    // Allocate and initialize host memory using vectors
    std::vector<float> h_A(h_N), h_B(h_N), h_C(h_N);
    
    for (int i = 0; i < h_N; ++i)
    {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    // Allocate Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);


    // Run the kernel in an infinite loop
    while (true) {
        // Zero output array
        cudaMemset(d_C, 0, bytes);
        
        // Launch kernel
        vectorAdd<<<1, 5>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        // cudaSyncho
        // Copy result back to host
        cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        
        // Print first element as a check
        std::cout << "C[0] = " << h_C[0] << " (expected 0)\n";
        std::cout << "C[1] = " << h_C[1] << " (expected 3)\n";
        std::cout << "C[2] = " << h_C[2] << " (expected 6)\n";
        
        // Sleep for 1 second
        sleep(1);
    }

    // Note: This code will never reach cleanup due to infinite loop
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
