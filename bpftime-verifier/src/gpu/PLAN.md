# GPU SIMT-Aware eBPF Verifier — Implementation Plan

## 1. Overview

This document describes the implementation plan for a **SIMT-aware GPU eBPF verifier** that
extends the existing PREVAIL-based CPU verifier with GPU-specific safety checks. The verifier
enforces the following invariants that cannot be checked by the standard eBPF verifier:

1. **Warp-uniform control flow**: Branch conditions and loop bounds must derive from warp-uniform values.
2. **No GPU-wide barriers**: `__syncthreads` (`bar.sync`) and similar primitives are prohibited.
3. **No non-uniform atomics on shared state**: Atomic operations on GPU-side map keys/values must use uniform addresses.
4. **Resource budgets per hook**: Bounded instruction count, helper calls, and memory operations.
5. **GPU helper type safety**: GPU helpers (501–511) must be called with correct argument types.
6. **GPU map type awareness**: Map types 1501–1527 must be recognized and validated.

## 2. Architecture

```
                    eBPF bytecode (loaded by nv_attach_impl)
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Phase 1: Standard PREVAIL  │  ← existing, extended with GPU
                │   - Memory safety            │     platform (helpers, maps, types)
                │   - Bounded loops            │
                │   - Helper type checking     │
                └──────────────┬──────────────┘
                               │ PASS
                               ▼
                ┌─────────────────────────────┐
                │   Phase 2: SIMT Safety Pass  │  ← NEW
                │   - Uniformity analysis      │
                │   - Barrier detection        │
                │   - Atomic safety            │
                │   - Resource budget          │
                └──────────────┬──────────────┘
                               │ PASS
                               ▼
                    PTX compilation (existing)
```

### File Structure

```
bpftime-verifier/
├── src/
│   ├── gpu/
│   │   ├── PLAN.md                    # this file
│   │   ├── gpu_verifier.hpp           # public API
│   │   ├── gpu_verifier.cpp           # main entry point, orchestrates phases
│   │   ├── gpu_platform.hpp           # GPU platform spec (helpers, maps, types)
│   │   ├── gpu_platform.cpp           # GPU-specific PREVAIL platform impl
│   │   ├── uniformity_analysis.hpp    # warp-uniform vs lane-varying analysis
│   │   ├── uniformity_analysis.cpp    # dataflow analysis implementation
│   │   ├── simt_safety_check.hpp      # barrier/atomic/divergence checks
│   │   ├── simt_safety_check.cpp      # pattern matching on instructions
│   │   ├── resource_budget.hpp        # per-hook resource budget definitions
│   │   └── resource_budget.cpp        # budget counting and enforcement
│   ├── bpftime-verifier.cpp           # existing (modify to add GPU path)
│   └── platform-impl.cpp             # existing (modify to support GPU types)
├── include/
│   └── bpftime-verifier.hpp           # existing (add GPU verify API)
└── test/
    ├── gpu_verifier_test.cpp          # NEW: unit tests for GPU verifier
    ├── gpu_uniformity_test.cpp        # NEW: uniformity analysis tests
    ├── gpu_safety_test.cpp            # NEW: SIMT safety check tests
    └── gpu_resource_budget_test.cpp   # NEW: resource budget tests
```

## 3. Detailed Design

### 3.1 GPU Platform Implementation (`gpu_platform.cpp`)

Extends PREVAIL's `ebpf_platform_t` interface to recognize GPU programs and types.

#### Program Types

| Section prefix | Program type | Context descriptor |
|---------------|-------------|-------------------|
| `cuda__` / `kprobe/` | `CUDA_PROBE` | `struct pt_regs` (same as kprobe, 168 bytes) |
| `kretprobe/` | `CUDA_RETPROBE` | `struct pt_regs` |

The program type for GPU sections should reuse the kprobe context descriptor from
`g_ebpf_platform_linux` since GPU eBPF programs receive `struct pt_regs *ctx`.

#### GPU Helper Prototypes

Register all GPU-specific helpers with correct argument types:

```cpp
// Helper 501: puts(const char *str) -> int
{501, {"bpf_puts", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM, EBPF_ARGUMENT_TYPE_CONST_SIZE,
        EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE}}}

// Helper 502: get_globaltimer() -> u64
{502, {"bpf_get_globaltimer", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_DONTCARE, ...}}}

// Helper 503: get_block_idx(u64 *x, u64 *y, u64 *z) -> int
{503, {"bpf_get_block_idx", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM,
        EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE}}}

// Helper 504: get_block_dim(u64 *x, u64 *y, u64 *z) -> int
{504, {"bpf_get_block_dim", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM,
        EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE}}}

// Helper 505: get_thread_idx(u64 *x, u64 *y, u64 *z) -> int
// → LANE-VARYING: return value is NOT warp-uniform
{505, {"bpf_get_thread_idx", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM,
        EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE}}}

// Helper 508: get_grid_dim(u64 *x, u64 *y, u64 *z) -> int
{508, {"bpf_get_grid_dim", EBPF_RETURN_TYPE_INTEGER,
       {EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM,
        EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM, EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE}}}

// Helper 509: get_sm_id() -> u64       [LANE-VARYING on some archs]
{509, {"bpf_get_sm_id", EBPF_RETURN_TYPE_INTEGER, {EBPF_ARGUMENT_TYPE_DONTCARE, ...}}}

// Helper 510: get_warp_id() -> u64     [WARP-UNIFORM]
{510, {"bpf_get_warp_id", EBPF_RETURN_TYPE_INTEGER, {EBPF_ARGUMENT_TYPE_DONTCARE, ...}}}

// Helper 511: get_lane_id() -> u64     [LANE-VARYING by definition]
{511, {"bpf_get_lane_id", EBPF_RETURN_TYPE_INTEGER, {EBPF_ARGUMENT_TYPE_DONTCARE, ...}}}
```

#### GPU Map Types

Map type 1501-1527 should map to the underlying standard types for PREVAIL's purposes:

| GPU Map Type | ID | Maps to | Notes |
|-------------|-----|---------|-------|
| `GPU_HASH_MAP` | 1501 | `BPF_MAP_TYPE_HASH` | Same key/value semantics |
| `PERGPUTD_ARRAY_MAP` | 1502 | `BPF_MAP_TYPE_ARRAY` | Per-thread, key = index |
| `GPU_ARRAY_MAP` | 1503 | `BPF_MAP_TYPE_ARRAY` | Single copy |
| `GPU_KERNEL_SHARED_ARRAY_MAP` | 1504 | `BPF_MAP_TYPE_ARRAY` | Kernel-shared |
| `PERGPUTD_ARRAY_HOST_MAP` | 1512 | `BPF_MAP_TYPE_ARRAY` | Host-backed per-thread |
| `GPU_ARRAY_HOST_MAP` | 1513 | `BPF_MAP_TYPE_ARRAY` | Host-backed shared |
| `GPU_RINGBUF_MAP` | 1527 | `BPF_MAP_TYPE_RINGBUF` | Ring buffer |

### 3.2 Uniformity Analysis (`uniformity_analysis.cpp`)

This is the **core novel contribution**. It performs abstract interpretation over the eBPF
instruction sequence to classify every register and stack slot as either **UNIFORM** (same
across all warp lanes) or **VARYING** (potentially different per lane).

#### Abstract Domain

```cpp
enum class Uniformity {
    UNIFORM,   // Value is identical across all lanes in a warp
    VARYING,   // Value may differ per lane
    UNKNOWN,   // Not yet analyzed (initial state for most registers)
};

struct UniformityState {
    Uniformity regs[11];        // R0-R10
    // R10 (frame pointer) is always UNIFORM
    // R1 (context pointer) is UNIFORM (same ctx for all lanes in warp-leader model)
};
```

#### Transfer Rules

1. **Constants/immediates**: `MOV64_IMM`, `MOV32_IMM` → UNIFORM
2. **Arithmetic (reg op reg)**:
   - UNIFORM op UNIFORM → UNIFORM
   - UNIFORM op VARYING → VARYING
   - VARYING op anything → VARYING
3. **Arithmetic (reg op imm)**: preserves source uniformity
4. **Helper call results**:
   - Most helpers return UNIFORM (e.g., `map_lookup`, `get_block_idx`, `get_globaltimer`, `get_warp_id`)
   - Lane-specific helpers return VARYING: `get_thread_idx` (505), `get_lane_id` (511)
   - `get_sm_id` (509) → conservatively VARYING (SM assignment could differ per warp in some configs)
5. **Memory load from map**: UNIFORM (maps are shared, same key → same value)
6. **Memory load from stack**: inherit uniformity of the stored value (tracked separately)
7. **Context load (from R1 + offset)**: UNIFORM (warp-leader model, all lanes see same ctx)
8. **Phi nodes (control flow merge)**: UNIFORM ∧ UNIFORM → UNIFORM, otherwise VARYING

#### Fixed-Point Iteration

For loops, iterate until the uniformity state at each program point reaches a fixed point.
Since the lattice has only 3 elements and is monotonic (UNKNOWN → UNIFORM/VARYING), this
converges in at most O(instructions × registers) steps.

#### Implementation Strategy

Operate on the raw eBPF instruction stream (`ebpf_inst` array) rather than PREVAIL's internal
`InstructionSeq`, because:
- We need this to work independently of PREVAIL's abstract interpretation
- Raw instructions are directly available from the loading path
- Simpler to implement and test

Build a basic-block CFG from the instruction stream:
1. Scan for jump targets to find basic block boundaries
2. For each basic block, compute transfer functions
3. Iterate to fixed point across the CFG

### 3.3 SIMT Safety Checks (`simt_safety_check.cpp`)

After uniformity analysis, perform the following checks:

#### Check 1: Warp-Uniform Branch Conditions

For every conditional jump instruction (`BPF_JEQ`, `BPF_JNE`, `BPF_JGT`, `BPF_JGE`,
`BPF_JLT`, `BPF_JLE`, `BPF_JSET`, `BPF_JSGT`, `BPF_JSGE`, `BPF_JSLT`, `BPF_JSLE`):

- The source register(s) must be UNIFORM
- If comparing reg-to-reg, both must be UNIFORM
- If comparing reg-to-imm, the register must be UNIFORM

**Rationale**: A VARYING branch condition causes warp divergence. In the warp-leader execution
model, only the leader executes the eBPF handler, so divergence within the handler is
semantically incorrect (the leader's branch outcome is applied to all lanes).

**Exception**: We could allow VARYING branches inside clearly-bounded regions that don't
affect shared state, but for the initial implementation, we reject all VARYING branches
(conservative but sound).

#### Check 2: No GPU-wide Barriers

Scan for helper calls that correspond to barrier operations:

- Helper 506 (`membar`/fence) — **REJECT** if used outside warp-leader context
- Any helper ID that maps to `__syncthreads`, `__threadfence_block`, etc.

In practice, the eBPF instruction set doesn't have native barrier instructions, so this is
primarily about rejecting specific helper IDs that map to barrier operations.

**Implementation**: Maintain a set of "prohibited helper IDs" for GPU programs. If any
`BPF_CALL` instruction references a prohibited helper, reject.

```cpp
static const std::set<int32_t> prohibited_gpu_helpers = {
    506,  // membar/fence — can cause deadlock in SIMT context
    // Add more as GPU helpers expand
};
```

#### Check 3: Atomic Safety

For `BPF_STX` with `BPF_ATOMIC` mode:
- The destination address (base register + offset) must be UNIFORM
- This prevents threads from atomically competing on different addresses in an uncoordinated way

For map update operations (`bpf_map_update_elem`, helper ID 2):
- The key pointer (R2) must point to UNIFORM data
- This ensures all lanes would produce the same map update (safe for warp-leader dedup)

#### Check 4: Map Update Key Uniformity

When `bpf_map_update_elem` (helper 2) is called:
- R1 (map pointer) must be UNIFORM (always true, comes from map fd)
- R2 (key pointer) must point to UNIFORM memory
- R3 (value pointer) — can be VARYING (value computed per-lane is aggregated by leader)
- R4 (flags) must be UNIFORM

### 3.4 Resource Budget (`resource_budget.cpp`)

Each GPU hook type has maximum allowed resource usage:

```cpp
struct GpuResourceBudget {
    uint32_t max_instructions;      // Total instructions (default: 4096)
    uint32_t max_helper_calls;      // Total helper invocations (default: 64)
    uint32_t max_memory_ops;        // Total load/store operations (default: 256)
    uint32_t max_map_lookups;       // Max bpf_map_lookup_elem calls (default: 32)
    uint32_t max_map_updates;       // Max bpf_map_update_elem calls (default: 16)
};
```

Default budgets per hook type:

| Hook Type | max_insn | max_helper | max_mem | max_lookup | max_update |
|-----------|----------|------------|---------|------------|------------|
| `cuda_probe` (entry) | 4096 | 64 | 256 | 32 | 16 |
| `cuda_retprobe` (exit) | 4096 | 64 | 256 | 32 | 16 |
| `memcapture` | 2048 | 32 | 128 | 16 | 8 |
| `directly_run` | 8192 | 128 | 512 | 64 | 32 |
| `scheduler` | 1024 | 16 | 64 | 8 | 4 |

**Implementation**: Single pass counting instructions, categorizing each, and checking against budget.

### 3.5 Public API (`gpu_verifier.hpp`)

```cpp
namespace bpftime {
namespace verifier {
namespace gpu {

/// Configuration for GPU verification
struct GpuVerifierConfig {
    GpuResourceBudget budget;       // Resource limits
    bool strict_uniformity = true;  // Reject all VARYING branches
    bool allow_membar = false;      // Allow helper 506 (membar)
};

/// Result of GPU verification
struct GpuVerifyResult {
    bool passed;
    std::string error_message;       // Empty if passed

    // Diagnostics
    uint32_t instruction_count;
    uint32_t helper_call_count;
    uint32_t memory_op_count;
    uint32_t varying_branch_count;   // 0 if passed
    uint32_t prohibited_helper_count;

    // Uniformity info (for debugging)
    std::vector<Uniformity> final_reg_uniformity;  // R0-R10 at exit
};

/// Main entry point: verify an eBPF program for GPU execution
GpuVerifyResult verify_gpu_program(
    const ebpf_inst *instructions,
    size_t num_instructions,
    const std::string &section_name,
    const GpuVerifierConfig &config = {}
);

/// Verify using raw uint64_t words (for compatibility with existing API)
GpuVerifyResult verify_gpu_program(
    const uint64_t *raw_inst,
    size_t num_inst,
    const std::string &section_name,
    const GpuVerifierConfig &config = {}
);

/// Get default budget for a hook type (determined by section name)
GpuResourceBudget get_default_budget(const std::string &section_name);

} // namespace gpu
} // namespace verifier
} // namespace bpftime
```

### 3.6 Integration with Existing Code

#### In `bpftime-verifier.cpp`

Add a GPU verification path:

```cpp
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
                                                size_t num_inst,
                                                const std::string &section_name)
{
    // Check if this is a GPU program
    if (is_gpu_section(section_name)) {
        auto result = gpu::verify_gpu_program(raw_inst, num_inst, section_name);
        if (!result.passed) {
            return result.error_message;
        }
        return {};  // passed
    }

    // Existing CPU verification path...
    // (unchanged)
}
```

#### In `platform-impl.cpp`

Extend `bpftime_get_program_type` and `bpftime_get_map_type` to handle GPU types:

```cpp
static EbpfProgramType bpftime_get_program_type(const std::string &section, ...)
{
    if (section.starts_with("uprobe") || ... ) {
        return g_ebpf_platform_linux.get_program_type(section, path);
    } else if (section.starts_with("kprobe/") || section.starts_with("kretprobe/") ||
               section.starts_with("cuda__")) {
        // GPU program — use kprobe type from Linux platform
        return g_ebpf_platform_linux.get_program_type("kprobe/placeholder", path);
    }
    throw ...;
}

static EbpfMapType bpftime_get_map_type(uint32_t platform_specific_type)
{
    // Handle GPU map types by mapping to their base types
    if (platform_specific_type >= 1500) {
        uint32_t base_type = platform_specific_type - 1500;
        // 1501 → HASH(1), 1502-1513 → ARRAY(2), 1527 → RINGBUF(27)
        if (base_type == 1) return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_HASH);
        if (base_type >= 2 && base_type <= 13) return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_ARRAY);
        if (base_type == 27) return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_RINGBUF);
    }
    // existing code...
}
```

#### In `nv_attach_impl.cpp`

Call verifier before PTX compilation:

```cpp
// In create_attach_with_ebpf_callback() or run_attach_entry_on_gpu():
#ifdef ENABLE_BPFTIME_VERIFIER
    auto verify_result = bpftime::verifier::gpu::verify_gpu_program(
        instructions, num_instructions, section_name);
    if (!verify_result.passed) {
        spdlog::error("GPU eBPF verification failed: {}", verify_result.error_message);
        return -EINVAL;  // or respect BPFTIME_VERIFIER_LEVEL
    }
#endif
```

## 4. Test Plan

### 4.1 Unit Tests: Uniformity Analysis (`gpu_uniformity_test.cpp`)

| Test Case | eBPF Program | Expected Uniformity at Exit |
|-----------|-------------|---------------------------|
| `const_is_uniform` | `MOV64_IMM R0, 42; EXIT` | R0 = UNIFORM |
| `thread_idx_is_varying` | `CALL 505; EXIT` | R0 = VARYING |
| `block_idx_is_uniform` | `CALL 503; EXIT` | R0 = UNIFORM |
| `lane_id_is_varying` | `CALL 511; EXIT` | R0 = VARYING |
| `warp_id_is_uniform` | `CALL 510; EXIT` | R0 = UNIFORM |
| `uniform_plus_uniform` | `MOV R1 42; MOV R2 10; ADD R1 R2; EXIT` | R1 = UNIFORM |
| `uniform_plus_varying` | `MOV R1 42; CALL 511; ADD R1 R0; EXIT` | R1 = VARYING |
| `varying_after_branch_merge` | uniform branch, both paths set R0 differently | R0 = UNIFORM (both uniform) |
| `map_lookup_uniform` | `map_lookup_elem(map, &key)` where key is uniform | R0 = UNIFORM |
| `map_lookup_from_varying_key` | key derived from thread_idx | key is VARYING (verifier should flag map_update but lookup is OK) |
| `loop_preserves_uniformity` | bounded loop with uniform counter | counter stays UNIFORM |

### 4.2 Unit Tests: SIMT Safety (`gpu_safety_test.cpp`)

| Test Case | eBPF Program | Expected Result |
|-----------|-------------|-----------------|
| `uniform_branch_ok` | `if (block_idx.x > 5)` | PASS |
| `varying_branch_rejected` | `if (thread_idx.x > 5)` | REJECT: "branch condition is lane-varying" |
| `varying_loop_rejected` | `for (i = 0; i < lane_id; i++)` | REJECT: "loop bound is lane-varying" |
| `barrier_rejected` | `CALL 506` (membar) | REJECT: "prohibited helper: membar" |
| `uniform_atomic_ok` | `XADD [R1+0], R2` where R1 is UNIFORM | PASS |
| `varying_atomic_rejected` | `XADD [R1+0], R2` where R1 is VARYING | REJECT: "atomic on varying address" |
| `map_update_uniform_key_ok` | `map_update(map, &uniform_key, &val, 0)` | PASS |
| `map_update_varying_key_rejected` | `map_update(map, &varying_key, &val, 0)` | REJECT: "map update key is lane-varying" |
| `simple_counter_ok` | cuda-counter example program | PASS |
| `threadhist_ok` | threadhist example program | PASS |
| `kernelretsnoop_ok` | kernelretsnoop example program | PASS |

### 4.3 Unit Tests: Resource Budget (`gpu_resource_budget_test.cpp`)

| Test Case | eBPF Program | Expected Result |
|-----------|-------------|-----------------|
| `within_budget` | 10 instructions, 1 helper call | PASS |
| `too_many_instructions` | 5000 instructions (budget=4096) | REJECT |
| `too_many_helpers` | 100 helper calls (budget=64) | REJECT |
| `too_many_map_updates` | 20 map_update calls (budget=16) | REJECT |
| `custom_budget_ok` | 5000 instructions with budget=8192 | PASS |

### 4.4 Integration Tests

| Test Case | Description |
|-----------|------------|
| `verify_all_examples` | Load every example/gpu/*.bpf.c program, verify they all pass |
| `verify_and_reject_bad` | Construct intentionally-bad GPU eBPF programs, verify rejection |
| `verify_then_compile` | Full pipeline: verify → compile to PTX → validate PTX |

## 5. Evaluation Plan (for OSDI/SOSP paper)

### RQ_V1: Verifier Correctness (Soundness & Completeness)

**Methodology**:
- Build a test suite of 30+ eBPF programs:
  - 15+ known-safe programs (from `example/gpu/` + paper's policy building blocks)
  - 15+ known-unsafe programs (constructed from `test-verify/examples/` patterns)
- Run each through the verifier and check for correct accept/reject decisions
- Report: true positives, true negatives, false positives, false negatives

**Expected results**:
- 100% true positive rate (all safe programs accepted)
- 100% true negative rate (all unsafe programs rejected)
- 0 false negatives (soundness guarantee)

**Presentation**: Table with program name, LOC, safety property tested, verifier decision, expected decision.

### RQ_V2: Verification Performance

**Methodology**:
- Measure wall-clock time for verification of programs from 16 to 4096 instructions
- Compare: (a) PREVAIL only, (b) PREVAIL + SIMT pass, (c) total load-to-ready time
- Run 1000 iterations for each, report median and P99

**Expected results**:
- SIMT pass adds < 1ms for typical programs (< 500 instructions)
- SIMT pass is < 10% of total PREVAIL verification time
- Total verification is << PTX compilation time (the bottleneck)

**Presentation**: Line chart: program size (x) vs verification time (y), two lines (PREVAIL, PREVAIL+SIMT).

### RQ_V3: Expressiveness

**Methodology**:
- Take all policy building blocks from the paper (Table support-matrix):
  - GPU L2 Stride Prefetch (45 LOC)
  - MaxSteals scheduler (16 LOC)
  - LatencyBudget scheduler (19 LOC)
  - kernelretsnoop (153 LOC)
  - threadhist (89 LOC)
  - launchlate (347 LOC)
- Verify each passes the GPU verifier
- Document any code modifications needed (should be 0)

**Expected results**: All existing GPU policies pass verification without modification.

**Presentation**: Table with policy name, LOC, verification result, modifications needed.

### RQ_V4: Safety Validation

**Methodology**:
- For each unsafe pattern from `test-verify/examples/`:
  1. Thread divergence (01): cascading if/else on thread_idx → warp serialization
  2. Non-coalesced memory (02): strided map access → bandwidth waste
  3. Atomic contention (03): all-thread atomicAdd on same address → serialization
  4. Deadlock (04): spinlock / producer-consumer / cross-lane dependency → GPU hang
  5. Bandwidth contention (05): excessive map writes → memory-bound conversion
- Show that the verifier rejects the dangerous patterns (1, 3, 4)
- Show that patterns 2, 5 are flagged as warnings (resource budget)
- Demonstrate what happens WITHOUT the verifier (GPU hang for pattern 4)

**Presentation**: Table with pattern, danger level, verifier action, consequence without verifier.

### RQ_V5: End-to-End Integration

**Methodology**:
- Run existing end-to-end workloads (llama.cpp, FAISS) with verifier enabled
- Measure: (a) policy load time with/without verifier, (b) runtime performance unchanged
- Confirm all policies pass verification and execute correctly

**Expected results**:
- Policy load time increases by < 1ms (negligible vs. CUDA kernel compilation)
- Runtime performance identical (verification is load-time only)

**Presentation**: Bar chart comparing load latency with/without verifier; confirmation table.

## 6. Implementation Order & Dependencies

```
Phase 1 (Foundation):
  Task 1: gpu_platform.cpp — GPU helper prototypes & map types
  Task 2: resource_budget.cpp — instruction/helper counting

Phase 2 (Core Analysis):
  Task 3: uniformity_analysis.cpp — CFG construction + dataflow analysis
  Task 4: simt_safety_check.cpp — checks using uniformity results

Phase 3 (Integration):
  Task 5: gpu_verifier.cpp — orchestrate Phase 1+2, public API
  Task 6: Modify platform-impl.cpp + bpftime-verifier.cpp for GPU path
  Task 7: Modify CMakeLists.txt to build new files + tests

Phase 4 (Testing):
  Task 8: Unit tests (uniformity, safety, budget)
  Task 9: Integration tests (existing examples pass)

Phase 5 (Evaluation):
  Task 10: Build unsafe test programs
  Task 11: Verification performance benchmark
  Task 12: End-to-end test with real workloads
```

## 7. Key Design Decisions & Rationale

### Why operate on raw eBPF instructions, not PREVAIL IR?

PREVAIL's `InstructionSeq` is an abstract representation optimized for the Crab abstract
interpreter. Adding a new abstract domain (uniformity) to Crab would require deep changes
to PREVAIL internals. Instead, we run PREVAIL first (unchanged) for memory safety, then
run our own lightweight pass on the raw instructions. This is:
- Simpler to implement and maintain
- Decoupled from PREVAIL version updates
- Easier to test in isolation

### Why conservative (reject all VARYING branches)?

The warp-leader execution model means only one lane runs the eBPF handler. A VARYING branch
in this context is semantically suspicious — it suggests the program is trying to do per-lane
logic, which contradicts the warp-leader model. If future use cases need per-lane branches
(e.g., for aggregation within the handler), we can add an `@allow_varying` annotation.

### Why not formal verification?

For OSDI/SOSP, empirical soundness (tested on comprehensive test suite) is sufficient.
Formal proofs (e.g., in Coq/Isabelle) would be a separate paper's contribution. The verifier
is small enough (~1-2 KLOC) that manual audit + thorough testing provides sufficient confidence.

## 8. Estimated Complexity

| Component | Estimated LOC | Difficulty |
|-----------|--------------|------------|
| `gpu_platform.cpp` | ~200 | Low |
| `resource_budget.cpp` | ~150 | Low |
| `uniformity_analysis.cpp` | ~500 | Medium-High |
| `simt_safety_check.cpp` | ~300 | Medium |
| `gpu_verifier.cpp` | ~150 | Low |
| Integration changes | ~100 | Low |
| Unit tests | ~800 | Medium |
| **Total** | **~2200** | |
