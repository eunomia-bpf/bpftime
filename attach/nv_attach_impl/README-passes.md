## NV PTX Passes - How to Run End-to-End

This guide demonstrates how to trigger the `nv_attach_impl` fatbin processing workflow on an existing CUDA example. The process involves extracting PTX, serially executing external passes, adding a trampoline, and repackaging the fatbin with `nvcc`.

### Prerequisites
- CUDA is installed (e.g., in `/usr/local/cuda-12.6`).
- `nvcc` and `cuobjdump` are available in your PATH.

### Build and Run (Automated Script)

```bash
# Optional: Specify the CUDA installation path
export BPFTIME_CUDA_ROOT=/usr/local/cuda-12.6

# Run the script (this will automatically build and run the server/client)
bash example/gpu/cuda-counter/run_nv_ptx_passes.sh
```

**Script Steps:**
- Builds `bpftime` with the flags `-DBPFTIME_ENABLE_CUDA_ATTACH=ON` and `-DBPFTIME_CUDA_ROOT`.
- Builds the `cuda_probe` and `vec_add` examples located in `example/gpu/cuda-counter`.
- Starts the server (with syscall-server preloaded).
- Runs `vec_add` with the agent preloaded, which triggers the CUDA fatbin registration.
- Inspects the output files `main.ptx` (after serial execution of external passes) and `out.fatbin` (the repackaged result) in the `/tmp/bpftime-recompile-nvcc` directory.

### Expected Results
- The `vec_add` program should run and produce its normal output; CUDA events should be observable in the server-side logs.
- The file `/tmp/bpftime-recompile-nvcc/main.ptx` should exist and contain the following:
  - A `.func __probe_func__<kernel>` definition.
  - A `call __probe_func__<kernel>;` instruction inside the target kernel's body.
  - If memory capture attach is enabled, a `.func __memcapture__N` definition and a corresponding `call __memcapture__N;` instruction will be present.
  - The file should **not** contain any temporary registers like `%ptxpass_*`, `mov.u64 %..., %globaltimer;` instructions, or any injected marker comments (as these should have been removed).
- The file `/tmp/bpftime-recompile-nvcc/out.fatbin` should exist and be non-empty (indicating it has been repackaged by `nvcc`).

### Troubleshooting
- If `/tmp/bpftime-recompile-nvcc/main.ptx` is not generated:
  - Verify that the following three executables exist:
    - `build/attach/nv_attach_impl/pass/ptxpass_kprobe_entry/ptxpass_kprobe_entry`
    - `build/attach/nv_attach_impl/pass/ptxpass_kretprobe/ptxpass_kretprobe`
    - `build/attach/nv_attach_impl/pass/ptxpass_kprobe_memcapture/ptxpass_kprobe_memcapture`
  - To drive the pass order and subset via JSON configuration, set the `BPFTIME_PTXPASS_DIR` environment variable to a directory containing `*.json` configuration files. The log should print "Discovered <N> pass definitions from <dir>".
  - If the log shows "Discovered 0 pass definitions...", the system will use a fallback process (which can still complete the injection and repackaging). Check the run logs to confirm that `hack_fatbin` has been executed.

### Pass Standalone Executable & Configuration (JSON-only I/O)
- **Path:** `attach/nv_attach_impl/pass/ptxpass_*`
  - All passes communicate via JSON-structured data through stdin/stdout (plain text PTX is no longer accepted):
    - **Input (stdin):**
      - `full_ptx`: string, the complete PTX to be processed.
      - `to_patch_kernel`: string, the name of the target kernel (optional).
      - `global_ebpf_map_info_symbol`: string, defaults to `map_info`.
      - `ebpf_communication_data_symbol`: string, defaults to `constData`.
      - Other pass-specific fields (e.g., `source_symbol`, `copy_bytes`, `align_bytes` for memcapture).
    - **Output (stdout):**
      - `output_ptx`: string, the transformed PTX (an empty string or omitted field indicates no modifications).
  - The `PTX_ATTACH_POINT` environment variable is passed to specify the current attachment point being processed.
  - **Note:** Stdin only accepts JSON. Non-JSON input will cause the program to error out and exit immediately.
- **Default JSON configurations** are located at: `attach/nv_attach_impl/configs/ptxpass/*.json` (these can be copied to a custom directory and specified via `BPFTIME_PTXPASS_DIR`).