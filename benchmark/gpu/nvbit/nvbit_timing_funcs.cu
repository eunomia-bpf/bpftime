/*
 * Device code to be injected by the NVBit minimal vector addition instrumentation tool
 */

#include <stdint.h>
#include <stdio.h>

/* Function to record the start time of kernel execution */
extern "C" __device__ __noinline__ void record_start_time(uint64_t time_ptr) {
    /* Only the first thread records the time */
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
    //     blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     /* Get the current clock time */
    //     uint64_t start_time = clock64();
    //     /* Store it in the provided memory location */
    //     *((uint64_t*)time_ptr) = start_time;
    // }
}

/* Function to record the end time of kernel execution */
extern "C" __device__ __noinline__ void record_end_time(uint64_t time_ptr) {
    /* Only the first thread records the time */
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
    //     blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     /* Get the current clock time */
    //     uint64_t end_time = clock64();
    //     /* Store it in the provided memory location */
    //     *((uint64_t*)time_ptr) = end_time;
    // }
} 