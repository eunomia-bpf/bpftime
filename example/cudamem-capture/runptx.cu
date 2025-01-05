// infinite_kernel.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void infiniteKernel()
{
    // Only thread (0,0) in block (0,0) will print once, then spin forever
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Infinite kernel started on device.\n");
    }
    // Prevent the compiler from optimizing the loop away
    while (true) {
        // Optionally, you can do a dummy operation:
    }
}

int main()
{
    // Launch 1 block of 1 thread
    infiniteKernel<<<1, 1>>>();
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel launched. Waiting for it to run...\n");
    // This will block indefinitely, since the kernel never returns
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n",
                err, cudaGetErrorString(err));
        return 1;
    }

    // Normally we’d never reach here
    printf("This line will never be printed.\n");
    return 0;
}
