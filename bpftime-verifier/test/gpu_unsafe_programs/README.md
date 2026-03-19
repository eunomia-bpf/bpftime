# GPU Unsafe eBPF Reference Programs

This directory contains reference GPU eBPF programs for the SIMT-aware verifier evaluation in `bpftime-verifier`. The files are written in realistic `.bpf.c` style and document the boundary between programs that a GPU-aware verifier should reject and programs that should pass.

These are primarily documentation and evaluation fixtures for the paper artifact. They do not need to compile with `clang` today; the goal is to make the intended verifier behavior concrete and easy to review.

## Purpose

Traditional eBPF verification checks memory safety, helper validity, and bounded execution. That is necessary but not sufficient for GPU execution. A SIMT-aware verifier must also reason about:

- warp-uniform control flow;
- warp-uniform side effects on shared state;
- prohibited synchronization helpers such as GPU fences;
- bounded resource usage per hook.

This suite captures those cases in a compact form for OSDI/SOSP evaluation.

## Files

| File | Expected result | Pattern |
| --- | --- | --- |
| `varying_branch.bpf.c` | REJECT | Branch predicate derived from `bpf_get_thread_idx()` (`505`), causing lane-varying control flow |
| `prohibited_helper.bpf.c` | REJECT | Calls prohibited GPU fence helper `bpf_gpu_membar()` (`506`) |
| `varying_atomic.bpf.c` | REJECT | Atomic update targets an address derived from `thread_idx` |
| `varying_map_key.bpf.c` | REJECT | `bpf_map_update_elem()` key pointer and key value derive from `lane_id` |
| `resource_exceeded.bpf.c` | REJECT | Static helper footprint exceeds the default CUDA/kprobe helper-call budget (`>64`) |
| `safe_counter.bpf.c` | PASS | Uses only warp-uniform predicates and a uniform atomic address |
| `safe_block_idx_branch.bpf.c` | PASS | Branches only on `bpf_get_block_idx()` (`503`), which is warp-uniform |

## How To Read These Programs

- Unsafe cases intentionally use helpers such as `bpf_get_thread_idx()` (`505`) and `bpf_get_lane_id()` (`511`) as sources of lane-varying state.
- Safe cases only branch on helpers that are uniform within a warp, such as `bpf_get_warp_id()` (`510`) and `bpf_get_block_idx()` (`503`).
- `resource_exceeded.bpf.c` spells out repeated helper invocations directly because the resource budget is a property of the program body, not just runtime loop iteration counts.

## Why This Matters

On CPUs, these programs would often look acceptable: branches are bounded, helpers are known, and memory accesses are within declared map values. On GPUs, the same patterns can cause warp divergence, serialization, or deadlock-prone behavior when the eBPF hook is injected into a CUDA kernel and executed under SIMT semantics.

These reference programs therefore serve three roles:

- paper-facing examples for the evaluation section;
- future lowering targets for bytecode-based verifier tests;
- artifact documentation showing why CPU-style verification is insufficient for GPU eBPF hooks.
