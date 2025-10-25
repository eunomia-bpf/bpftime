## NV PTX Passes - 如何运行端到端

本说明演示如何在现有 CUDA 例子上触发 nv_attach_impl 的 fatbin 处理流程（提取 PTX -> 串行执行外部 Pass -> 加 trampoline -> nvcc 重打包 fatbin）。

### 先决条件
- 已安装 CUDA（例如 `/usr/local/cuda-12.6`）。
- 可用的 `nvcc` 和 `cuobjdump`。

### 构建与运行（自动脚本）

```bash
# 可选：指定 CUDA 路径
export BPFTIME_CUDA_ROOT=/usr/local/cuda-12.6

# 运行脚本（会自动构建并运行 server/client）
bash example/gpu/cuda-counter/run_nv_ptx_passes.sh
```

脚本步骤：
- 使用 `-DBPFTIME_ENABLE_CUDA_ATTACH=ON` 和 `-DBPFTIME_CUDA_ROOT` 构建 bpftime
- 构建 `example/gpu/cuda-counter` 下的 `cuda_probe` 和 `vec_add`
- 启动 server（syscall-server 预加载）
- 以 agent 预加载方式运行 `vec_add`，触发 CUDA fatbin 注册
- 在 `/tmp/bpftime-recompile-nvcc` 下检查 `main.ptx`（已串行执行外部 pass）、`out.fatbin`（重打包结果）

### 预期结果
- `vec_add` 正常运行输出；server 端日志可观察 CUDA 事件。
- `/tmp/bpftime-recompile-nvcc/main.ptx` 存在，且包含注入标记与指令：
  - `// __ptxpass_entry_injected__`、`// __ptxpass_ret_injected__`、`// __ptxpass_memcapture_injected__`
  - 专用寄存器与 `mov.u64 %..., %globaltimer;` 指令
- `/tmp/bpftime-recompile-nvcc/out.fatbin` 存在，显示非零大小（已用 nvcc 重打包）。

### 失败排查
- 若未生成 `/tmp/bpftime-recompile-nvcc/main.ptx`：
  - 检查 `passes.default.json` 路径是否为 `attach/nv_attach_impl/configs/ptxpass/passes.default.json`
  - 检查三个可执行是否存在：
    - `build/attach/nv_attach_impl/pass/ptxpass_kprobe_entry/ptxpass_kprobe_entry`
    - `build/attach/nv_attach_impl/pass/ptxpass_kretprobe/ptxpass_kretprobe`
    - `build/attach/nv_attach_impl/pass/ptxpass_kprobe_memcapture/ptxpass_kprobe_memcapture`
  - 查看运行日志，确认 `hack_fatbin` 已执行

### Pass 独立可执行与配置（JSON-only I/O）
- 路径：`attach/nv_attach_impl/pass/ptxpass_*`
  - 所有 pass 通过 stdin/stdout 进行 JSON 结构通信（不再接受纯文本 PTX）：
    - 输入（stdin）：
      - `full_ptx`: string，完整待处理 PTX
      - `to_patch_kernel`: string，目标 kernel 名（可选）
      - `global_ebpf_map_info_symbol`: string，默认 `map_info`
      - `ebpf_communication_data_symbol`: string，默认 `constData`
      - 其他 pass 特定字段（如 memcapture 的 `source_symbol`、`copy_bytes`、`align_bytes` 等）
    - 输出（stdout）：
      - `output_ptx`: string，变换后的 PTX（空字符串或缺省表示不修改）
  - `PTX_ATTACH_POINT` 环境变量传入当前处理的 attach point
- 默认 JSON 配置位于：`attach/nv_attach_impl/configs/ptxpass/*.json`
- 编排顺序配置：`attach/nv_attach_impl/configs/ptxpass/passes.default.json`




