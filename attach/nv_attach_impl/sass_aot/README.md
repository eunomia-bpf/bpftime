# SASS AOT spike — verified standalone live execution + companion
# interposition boundary (2026-09-05)

This directory contains a spike that compiles a real eBPF program to NVIDIA
SASS (cubin) and executes it on a GPU. On 2026-09-05 this path was verified
end-to-end on a live device. A second milestone adds a documented host-side
module interposition boundary that runs the BPF-derived SASS inside a host
application's own CUDA context, next to the application's own PTX-free SASS
module.

## Pipeline

1. **Real BPF ELF section** — a clang `-target bpf` object
   (`sass_aot_spike.bpf.o`) with section `cuda__/sass_aot` is loaded and its
   instruction words are extracted by `load_bpf_program_words`.
2. **Strict GPU verifier** — the program is checked with an explicit 8-byte
   PREVAIL GPU context (R1 points at a `uint64_t` read/write context) plus SIMT
   safety (warp-uniform branch conditions). Any program not proven safe is
   rejected before compilation.
3. **ptxpass NVPTX compiler** — the existing
   `ptxpass::compile_ebpf_to_ptx_from_words` backend translates the verified
   eBPF to PTX.
4. **ptxas** — `ptxas -arch=sm_120` lowers the PTX to a cubin (SASS) artifact.
5. **CUDA Driver API execution** — module load, function lookup, a 1x1x1 grid
   launch, `cuCtxSynchronize`, and a device-to-host copy back into the context.

## Verified live run

The spike BPF program (`SEC("cuda__/sass_aot")`) writes the value **42** to its
context. The standalone acceptance executable ran:

```
build-spike/attach/nv_attach_impl/sass_aot/bpftime_sass_aot_live \
  /tmp/bpftime_sass_aot_live-20260905 0
```

It exited zero and printed:

```
verified SASS result: 42
```

Post-run device state: driver **575.57.08**, **15 MiB** memory in use, **zero
percent** utilization, **P0** performance state.

## Companion/interposition boundary (2026-09-05)

The standalone path above proves the BPF-derived cubin runs, but not that it
can coexist with an existing application. The companion milestone adds a
minimal, documented host-side module interposition boundary:

- `companion_app_kernel.ptx` is a small host-application kernel, assembled by
  the build with `ptxas -arch=sm_120` into a **SASS-only, PTX-free cubin**
  (mirroring a shipping application that embeds precompiled SASS, no PTX).
- `bpftime_sass_aot_companion_live` is the acceptance executable. It:
  1. verifies and AOT-compiles the real BPF program exactly as the standalone
     path does (strict verifier first; a rejected program never reaches the
     boundary),
  2. creates the application's CUDA context, loads the application's own
     PTX-free module, and launches its own kernel (original execution, in
     flight),
  3. performs a negative check that the BPF-derived entry is **not** present
     in the application's own module (runtime evidence against rewriting),
  4. crosses the documented interposition boundary
     `bpftime::attach::sass_aot::execute_sass_aot_in_context(context,
     compiled, context_data)`, which loads the BPF-derived cubin into the
     *same* context as a **companion module**, launches the BPF entry, and
     synchronizes the context,
  5. reads back both results: the application's own result must still be
     correct (original execution preserved) and the BPF entry must have run.
- The boundary is fail-closed: inputs are validated before any CUDA call, and
  a verifier-rejected program is never loaded. `bpftime_sass_aot_tests`
  covers these rejections plus the PTX-free property of the application
  cubin, on the CPU without a GPU.

## Verified companion live run (2026-09-05)

On the RTX 5090 with the stock driver **575.57.08** (kernel modules
`nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_drm`; no kernel-module
swaps), the companion acceptance executable ran:

```sh
build-spike/attach/nv_attach_impl/sass_aot/bpftime_sass_aot_companion_live \
  /tmp/bpftime_sass_aot_companion-20260905 0
```

It exited zero and printed:

```
companion application result: 7 (original execution preserved)
bpf-derived companion entry result: 42
scope: companion SASS module in the application CUDA context; no application SASS rewriting
```

The run implicitly verified, at runtime, that the application's own module
does **not** contain the BPF-derived entry `sass_aot_probe` (the negative
`cuModuleGetFunction` check passes only when the lookup is
`CUDA_ERROR_NOT_FOUND`). `cuobjdump -sass` on the application cubin reports
only `companion_app_kernel` (sm_120 SASS); `cuobjdump -ptx` on it prints
nothing, proving the application artifact is PTX-free. Post-run driver state:
**575.57.08**, **15 MiB**, **zero percent** utilization, **P8**.

The standalone acceptance executable
(`bpftime_sass_aot_live`) was re-run the same day and still printed
`verified SASS result: 42`, so the companion milestone did not regress the
standalone path.

## Exact scope record

- **What this is**: a *companion/interposed SASS module*. The BPF-derived
  cubin is a second `CUmodule` in the application's active `CUcontext`,
  invoked through one documented host-side boundary function. The
  application's own binary, module, and SASS are untouched.
- **What this is not**: not arbitrary in-place SASS rewriting of an existing
  application, not a fatbin binary patch, not NVBit-style instruction
  patching, and not a real driver-API interposition (LD_PRELOAD/fatbin
  registrar) wired into a third-party process. The boundary is called by the
  host application at a documented lifecycle point; wiring it into a real
  module-load hook is the next step.
- **Helper/map limitations**: the BPF entry sees only its explicit verified
  context (an 8-byte `uint64_t` in the spike). It has no eBPF helper calls,
  no maps, and no access to the application's device memory or modules.
- **Performance status**: not measured. This is a functional integration
  demonstration, not performance evidence.

## Build and test

Build targets (with `-DBPFTIME_ENABLE_SASS_AOT_SPIKE=1`, which requires
`BPFTIME_ENABLE_CUDA_ATTACH=1` and `ENABLE_EBPF_VERIFIER=1`):

- `bpftime_verifier_tests`
- `bpftime_sass_aot_tests`
- `bpftime_sass_aot_live`
- `bpftime_sass_aot_companion_live`

Outcomes:

- The focused explicit-context verifier test passed all **3 assertions**: the
  strict verifier accepts an 8-byte context write, the same 8-byte write is
  rejected with a 7-byte context, and the write is rejected by the no-context
  verifier.
- The **CTest verifier suite** and the **SASS AOT suite** both passed.
- An invalid **lane-varying SIMT program is rejected** by the verifier before
  PTX is created, before a cubin exists, and before `ptxas` is invoked
  (verified with a fake-ptxas marker).

## Notes

- A local **Qwen 27B** model diagnosed and fixed a missing **R0 initialization**
  in the verifier unit fixture. An earlier, broader Qwen review timed out
  without findings.

## Limitations

- The standalone executable executes a **standalone generated cubin only**;
  it does not touch any application.
- The companion executable adds a **companion/interposed SASS module** in the
  application's own context, invoked through the documented boundary function.
  It does **not** perform arbitrary in-place SASS rewriting of an existing
  application, does not patch a fatbin binary, and is not a real
  driver-API/fatbin interposition wired into a third-party process.
- The BPF entry has no eBPF helpers and no maps; it sees only its explicit
  verified context (8 bytes in the spike).
- It is **not a paper-level benchmark**; it is a functional end-to-end
  acceptance demonstration. Performance has not been measured.
