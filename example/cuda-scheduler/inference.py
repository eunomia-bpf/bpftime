import torch
import time

from torch.utils.cpp_extension import load_inline
MAX_PIDS = 1024
fairshare = torch.zeros(MAX_PIDS, dtype=torch.int32,
                        device='cuda').pin_memory().to('cuda')
# launches = b.get_table("launch_count")
# active   = b.get_table("active_pids")
cuda_src = r'''
extern "C" {
__device__ int get_pid() {
    // 假设一个 block 对应一个 PID，并把 PID 传给 blockIdx.x
    return blockIdx.x;
}

// 全局数组，由 Host 更新
extern __device__ int fairshare[];

__device__ void scheduler_init(int pid, int *slice_rem) {
    *slice_rem = atomicAdd(&fairshare[pid], 0);
}

__device__ void scheduler_maybe_yield(int *slice_rem) {
    if (atomicSub(slice_rem, 1) <= 0) {
        // 简单 spin-yield 等待下一窗口
        while (atomicAdd(slice_rem, 0) <= 0) { /* busy-wait */ }
    }
}

__global__ void fair_kernel(float *data, float *weight, float *out,
                            int N, int *fs)
{
    int pid = get_pid();
    int rem = 0;
    scheduler_init(pid, &rem);

    // 简单模拟训练计算
    int idx = threadIdx.x;
    float acc = 0.0f;
    for (int i=0; i<N; i+=blockDim.x) {
        acc += data[pid*N + i + idx] * weight[idx];
        // 每做 32 次计算点，就做一次 yield 检查
        if ((i/32) % 1 == 0) {
            scheduler_maybe_yield(&rem);
        }
    }
    out[pid] = acc;
}
}
'''

module = load_inline(
    name="fairshare",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=["fair_kernel"],
    verbose=False
)

# 5) 准备模拟“多进程”数据：用不同 block 模拟不同 PID 的训练
num_procs = 4
N = 1024
data   = torch.randn(num_procs, N, device='cuda')
weight = torch.randn(N, device='cuda')
out    = torch.zeros(num_procs, device='cuda')

print("启动带公平份额调度的 CUDA kernel …")
module.fair_kernel(
    grid=(num_procs,1,1), block=(256,1,1),
    args=[data.data_ptr(),
          weight.data_ptr(),
          out.data_ptr(),
          N,
          fairshare.data_ptr()]
)

print("结果：", out)