import torch
import time

from torch.utils.cpp_extension import load_inline
MAX_PIDS = 1024

# Create fairshare tensor on CPU first, then pin memory, then move to GPU
fairshare = torch.zeros(MAX_PIDS, dtype=torch.int32, device='cpu').pin_memory().to('cuda')

cuda_src = r'''
extern "C" {
__device__ int get_pid() {
    return blockIdx.x;
}

// Global array for fair share scheduling
extern __device__ int fairshare[];

__device__ void scheduler_init(int pid, int *slice_rem) {
    *slice_rem = atomicAdd(&fairshare[pid], 0);
}

__device__ void scheduler_maybe_yield(int *slice_rem) {
    if (atomicSub(slice_rem, 1) <= 0) {
        // Simple spin-yield wait for next time slice
        while (atomicAdd(slice_rem, 0) <= 0) { /* busy-wait */ }
    }
}

__global__ void fair_kernel(float *data, float *weight, float *out,
                          int N, int *fs)
{
    int pid = get_pid();
    int rem = 0;
    scheduler_init(pid, &rem);

    // Simulate training computation
    int idx = threadIdx.x;
    float acc = 0.0f;
    for (int i=0; i<N; i+=blockDim.x) {
        acc += data[pid*N + i + idx] * weight[idx];
        // Check for yield every 32 computations
        if ((i/32) % 1 == 0) {
            scheduler_maybe_yield(&rem);
        }
    }
    out[pid] = acc;
}
}
'''

# Load CUDA module
module = load_inline(
    name="fairshare",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=["fair_kernel"],
    verbose=False
)

def run_inference(num_procs=4, N=1024):
    # Prepare data for multiple processes (simulated by different blocks)
    data = torch.randn(num_procs, N, device='cuda')
    weight = torch.randn(N, device='cuda')
    out = torch.zeros(num_procs, device='cuda')

    print(f"Starting CUDA kernel with fair share scheduling for {num_procs} processes...")
    
    # Launch kernel
    module.fair_kernel(
        grid=(num_procs,1,1), 
        block=(256,1,1),
        args=[data.data_ptr(),
              weight.data_ptr(),
              out.data_ptr(),
              N,
              fairshare.data_ptr()]
    )

    # Synchronize and print results
    torch.cuda.synchronize()
    print("Results:", out)
    
    # Clean up
    del data, weight, out
    torch.cuda.empty_cache()

def main():
    # Test with different numbers of processes
    process_counts = [2, 4, 8]
    for num_procs in process_counts:
        print(f"\nTesting with {num_procs} processes:")
        run_inference(num_procs=num_procs)
        
        # Clean up between runs
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()