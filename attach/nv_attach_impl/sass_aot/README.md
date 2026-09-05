# SASS AOT spike — verified standalone live execution (2026-09-05)

This directory contains a spike that compiles a real eBPF program to NVIDIA
SASS (cubin) and executes it on a GPU. On 2026-09-05 this path was verified
end-to-end on a live device.

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

## Build and test

Build targets (with `-DBPFTIME_ENABLE_SASS_AOT_SPIKE=1`, which requires
`BPFTIME_ENABLE_CUDA_ATTACH=1` and `ENABLE_EBPF_VERIFIER=1`):

- `bpftime_verifier_tests`
- `bpftime_sass_aot_tests`
- `bpftime_sass_aot_live`

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

- This spike executes a **standalone generated cubin only**. It does **not**
  perform arbitrary PTX-free instrumentation/injection of an application
  binary, and it is **not connected** to an existing application's SASS/fatbin.
- It is **not a paper-level benchmark**; it is a functional end-to-end
  acceptance demonstration.
