# GPU Plain Array Map 示例

本示例演示 `BPF_MAP_TYPE_GPU_ARRAY_MAP`（普通 array，非 per-thread）在用户态配合 bpftime 的行为。当前演示脚本采用 HOST 写回路径进行验证。

- BPF 程序：`gpu-plain-array.bpf.c` 在 `kretprobe/_Z9vectorAddPKfS0_Pf` 处对 key=0 自增。
- Host 程序：`gpu-plain-array.c` 周期性读取 `counter[0]` 打印。
- CUDA 程序：`vec_add.cu` 周期运行，触发 BPF 程序更新 map。

## 先决条件
- 已安装 clang/llvm、make、gcc。
- 可用的 CUDA 环境（`nvcc`、`libcudart`），且 GPU 可用。
- 仓库子模块 `third_party/libbpf`、`third_party/bpftool` 可用（示例 Makefile 会自动构建 bootstrap bpftool 与静态 libbpf）。
- 推荐：顶层已构建 bpftime 使能 CUDA attach：
  - `cmake -B build -S . -DBPFTIME_ENABLE_CUDA_ATTACH=ON && cmake --build build -j`

## 构建
```bash
cd example/gpu/gpu-plain-array
make
```
生成：
- `gpu-plain-array`（host 程序）
- `vec_add`（CUDA 测试内核）
- `.output/`（中间产物与 BPF skeleton）

## 运行（手工）
- 终端 A：
```bash
./vec_add
```
- 终端 B：
```bash
./gpu-plain-array
```
预期：`gpu-plain-array` 每秒打印 `counter[0]=N`，值单调递增；停止 `vec_add` 后，计数停止增长。

## 一键脚本
可使用本目录脚本快速验证（带超时与输出检查）：
```bash
bash ./run_demo.sh
```
脚本会：
- 构建示例（如未构建）。
- 启动 `vec_add`（10s 超时）。
- 并行运行 `gpu-plain-array`（8s 超时），采样两次计数并校验递增。
- 打印“PASS/FAIL”并输出日志到 `demo_out.txt`。

> 说明：脚本固定使用 HOST 写回路径（内部强制 `HOST_WRITEBACK=1`），确保在不同环境下稳定通过。

## 一致性说明（简述）
- 写入：单 key 自旋锁，写后 `__threadfence_system()`，保证写入对其他线程/主机可见。
- 读取：无锁，可能读到写入前的旧值，但不会读到部分写入。
- 如需“更强一致读”，可在读取侧加锁或做双读版本校验（本示例未启用）。

## Troubleshooting
- 无 `nvcc`：脚本会提示跳过 CUDA 构建；示例无法触发计数递增。
- 权限/依赖：确认已安装 clang、make、gcc，且可从 `third_party/` 构建 libbpf/bpftool。
- CUDA 驱动：需可运行最小 CUDA kernel；否则 `vec_add` 会失败。


