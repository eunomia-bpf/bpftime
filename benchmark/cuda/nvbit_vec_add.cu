/*
 * NVBit minimal instrumentation tool for vector addition benchmark
 * This is a simple probe that counts kernel calls and measures kernel execution time
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unordered_set>
#include <vector>

/* Path to NVBit headers - adjust as needed */
#include "../../../nvbit_release_x86_64/core/nvbit.h"
#include "../../../nvbit_release_x86_64/core/nvbit_tool.h"
#include "../../../nvbit_release_x86_64/core/utils/utils.h"

/* kernel counter */
uint32_t kernel_count = 0;

/* total execution time */
uint64_t total_exec_time_ns = 0;

/* Timestamps for measuring kernel execution time */
__managed__ uint64_t kernel_start_time = 0;
__managed__ uint64_t kernel_end_time = 0;

/* Mutex to prevent multiple kernels running concurrently */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed when the tool is loaded */
void nvbit_at_init() {
    /* Make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* Initialize mutex */
    pthread_mutex_init(&mutex, NULL);

    printf("NVBit: Minimal Vector Addition Instrumentation Tool\n");
    printf("------------------------------------------------\n");
}

/* Set to track instrumented functions */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Check if function was already instrumented */
    if (!already_instrumented.insert(func).second) {
        return;
    }

    /* Get the function's instructions */
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);
    
    /* Only instrument if there are instructions */
    if (instrs.empty()) {
        return;
    }
    
    /* Instrument the first instruction to record start time */
    Instr *first_instr = instrs.front();
    nvbit_insert_call(first_instr, "record_start_time", IPOINT_BEFORE);
    nvbit_add_call_arg_const_val64(first_instr, (uint64_t)&kernel_start_time);
    
    /* Instrument the last instruction to record end time */
    Instr *last_instr = instrs.back();
    nvbit_insert_call(last_instr, "record_end_time", IPOINT_AFTER);
    nvbit_add_call_arg_const_val64(last_instr, (uint64_t)&kernel_end_time);
}

/* Callback for CUDA driver events */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                          const char *name, void *params, CUresult *pStatus) {
    /* Only handle kernel launches */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* Get the function being launched */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* On kernel launch entry */
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);
            nvbit_enable_instrumented(ctx, func, true);
            kernel_start_time = 0;
            kernel_end_time = 0;
        } else {
            /* On kernel launch exit */
            CUDA_SAFECALL(cudaDeviceSynchronize());
            
            /* Increment kernel count */
            kernel_count++;
            
            /* Get kernel name */
            const char* kernel_name = nvbit_get_func_name(ctx, func, 0);
            
            /* Calculate execution time */
            if (kernel_end_time > kernel_start_time) {
                uint64_t exec_time = kernel_end_time - kernel_start_time;
                total_exec_time_ns += exec_time;
                
                /* Print timing information */
                printf("NVBit: Kernel %s - Time: %.3f us\n", 
                       kernel_name, exec_time / 1000.0f);
            }
            
            pthread_mutex_unlock(&mutex);
        }
    }
}

/* Callback for application termination */
void nvbit_at_term() {
    /* Print summary */
    printf("\nNVBit Instrumentation Summary:\n");
    printf("Total kernel calls: %u\n", kernel_count);
    printf("Total execution time: %.3f ms\n", total_exec_time_ns / 1000000.0f);
    if (kernel_count > 0) {
        printf("Average kernel time: %.3f us\n", 
               (total_exec_time_ns / 1000.0f) / kernel_count);
    }
} 