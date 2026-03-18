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

### Pass Host ABI And Metadata JSON
- **Path:** `attach/nv_attach_impl/pass/ptxpass_*`
  - The shared-library entry point is:

    ```cpp
    int process_input(const char *ptx_text, size_t ptx_len,
                      const char *meta_json, int meta_len,
                      char *output, int output_len);
    ```

  - `ptx_text`/`ptx_len` carries the raw PTX blob directly and is not JSON-encoded.
  - `meta_json` contains only small request metadata:
    - `to_patch_kernel`
    - `global_ebpf_map_info_symbol` (defaults to `map_info`)
    - `ebpf_communication_data_symbol` (defaults to `constData`)
    - `ebpf_instructions`
  - The JSON response always includes `modified`; `output_ptx` is included only when the pass actually rewrites PTX.
  - Legacy callers that still embed `input.full_ptx` in the metadata JSON can still be parsed by the shared core helpers, but the framework no longer emits that format.
- **Default JSON configurations** are located at: `attach/nv_attach_impl/configs/ptxpass/*.json` (these can be copied to a custom directory and specified via `BPFTIME_PTXPASS_DIR`). 
