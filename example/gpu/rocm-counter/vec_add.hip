#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <vector>

/*
hipcc vectorAdd.cpp -o vectorAdd
 */

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
    // Set vector size in constant memory
    const int h_N = 1 << 20;  // 1M elements
    hipMemcpyToSymbol(HIP_SYMBOL(d_N), &h_N, sizeof(h_N), 0, hipMemcpyHostToDevice);
    
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
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    // Copy to device
    hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice);

    // Set up execution parameters
    int threads = 1;
    int blocks = (h_N + threads - 1) / threads;

    // Run the kernel in an infinite loop
    while (true) {
        // Zero output array
        hipMemset(d_C, 0, bytes);
        
        // Launch kernel
        hipLaunchKernelGGL(vectorAdd, dim3(1), dim3(1), 0, 0, d_A, d_B, d_C);
        hipDeviceSynchronize();
        
        // Copy result back to host
        hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost);
        
        // Print first element as a check
        std::cout << "C[0] = " << h_C[0] << " (expected 0)\n";
        std::cout << "C[1] = " << h_C[1] << " (expected 3)\n";
        
        // Sleep for 1 second
        sleep(1);
    }

    // Note: This code will never reach cleanup due to infinite loop
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
