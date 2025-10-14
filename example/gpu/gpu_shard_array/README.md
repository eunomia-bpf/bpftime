# GPU Shard Array Map 示例

本示例演示 `BPF_MAP_TYPE_GPU_ARRAY_MAP`（普通 array，非 per-thread，单副本共享）在用户态配合 bpftime 的行为。示例依赖 agent 侧更新，server 仅读取，不做 HOST 写回。

- BPF 程序：`gpu_shard_array.bpf.c` 在 `kretprobe/vectorAdd` 处对 key=0 自增。
- Host 程序：`gpu_shard_array.c` 周期性读取 `counter[0]` 打印。
- CUDA 程序：`vec_add.cu` 周期运行，触发 BPF 程序更新 map。

## 先决条件
- 已安装 clang/llvm、make、gcc。
- 可用的 CUDA 环境（`nvcc`、`libcudart`），且 GPU 可用。
- 仓库子模块 `third_party/libbpf`、`third_party/bpftool` 可用（示例 Makefile 会自动构建 bootstrap bpftool 与静态 libbpf）。
- 推荐：顶层已构建 bpftime（启用 CUDA attach）：
  - `cmake -B build -S . -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda`
  - `cmake --build build -j`

## 构建
```bash
cd example/gpu/gpu_shard_array
make
```
生成：
- `gpu_shard_array`（host 程序）
- `vec_add`（CUDA 测试内核）
- `.output/`（中间产物与 BPF skeleton）

## 运行步骤（agent/server 模式）
1) 启动 server（只读进程）：
```bash
/root/bpftime_sy03/bpftime/build/tools/cli/bpftime load /root/bpftime_sy03/bpftime/example/gpu/gpu_shard_array/gpu_shard_array
```
2) 启动 agent 目标程序（加载 CUDA attach，触发 BPF）：
```bash
LD_PRELOAD=/root/bpftime_sy03/bpftime/build/runtime/agent/libbpftime-agent.so \
  BPFTIME_LOG_OUTPUT=console SPDLOG_LEVEL=debug \
  /root/bpftime_sy03/bpftime/example/gpu/gpu_shard_array/vec_add
```
3) 预期输出：
- server：周期打印 `counter[0]=N` 并单调递增。
- agent：可见 `CUDA Received call request id ...` 与 `CUDA: MAP_UPDATE success ...`。



## 一致性说明（简述）
- 写入：单 key 自旋锁，写后 `__threadfence_system()`，保证写入对其他线程/主机可见。
- 读取：无锁，可能读到写入前的旧值，但不会读到部分写入。
- 如需“更强一致读”，可在读取侧加锁或做双读版本校验（本示例未启用）。

## Troubleshooting
- 无 `nvcc`：脚本会提示跳过 CUDA 构建；示例无法触发计数递增。
- 权限/依赖：确认已安装 clang、make、gcc，且可从 `third_party/` 构建 libbpf/bpftool。
- CUDA 驱动：需可运行最小 CUDA kernel；否则 `vec_add` 会失败。


