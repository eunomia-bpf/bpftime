# GPU SIMT-Aware Verifier Improvement Plan

Date: 2026-03-18

## Reality Check

The current code is not a thin PREVAIL extension. It is a two-stage design:

1. `bpftime-verifier` calls PREVAIL for standard eBPF verification.
2. The GPU verifier then runs separate GPU-only passes over raw `ebpf_inst*`:
   - `src/gpu/resource_budget.cpp`
   - `src/gpu/uniformity_analysis.cpp`
   - `src/gpu/simt_safety_check.cpp`

That separation is the main architectural reason the paper-facing gaps exist. PREVAIL already knows a lot about pointer types, stack cells, helper effects, and loop bounds, but the GPU passes currently do not consume that information.

There is one important nuance: the current GPU uniformity pass is not completely register-only. It already tracks:

- per-register uniformity,
- 512 stack bytes,
- simple stack-pointer offsets, and
- a limited set of helper out-parameter writes.

So gap 1 is not literally "zero stack sensitivity." The real problem is that the implementation is only partially memory-sensitive and is still unsound/incomplete for general pointees, helper effects, and side-effect arguments.

## Current Test Status In This Workspace

I rebuilt and reran the GPU tests in the current workspace.

- `bpftime_gpu_verifier_tests`: pass
- `bpftime_gpu_verifier_e2e_tests`: pass

Rebuild command used:

```sh
cmake --build build --target bpftime_gpu_verifier_tests bpftime_gpu_verifier_e2e_tests -j2
```

Executables run:

```sh
build/bpftime-verifier/bpftime_gpu_verifier_tests --reporter compact
build/bpftime-verifier/bpftime_gpu_verifier_e2e_tests --reporter compact
```

Observed outputs:

- unit tests: `All tests passed (25 assertions in 4 test cases)`
- e2e tests: `All tests passed (30 assertions in 4 test cases)`

However, top-level `ctest -N` reports `Total Tests: 0`. The immediate artifact issue is not failing GPU tests; it is that root-level CTest discovery is not enabled. There is no top-level `enable_testing()`, so `add_test(...)` in `bpftime-verifier/CMakeLists.txt` is not visible from the root build. That is artifact friction a reviewer can trip over.

Also note that `bpftime-verifier/test/gpu_example_analysis.md` is stale relative to the current code. It claims helper out-parameter propagation is missing, but the current `uniformity_analysis.cpp` does implement a limited version of it, and the e2e tests rely on that.

## Key Architectural Findings From PREVAIL

### What PREVAIL already has

- A real abstract interpreter over a CFG built from `InstructionSeq`.
- A relational numeric domain (`SplitDBM` through `AddBottom`).
- A type domain in `crab::ebpf_domain_t::TypeDomain`.
- Stack memory modeling in `crab::domains::array_domain_t`.
- Per-stack-cell parallel facts for:
  - `types`
  - `svalues`
  - `uvalues`
  - `ctx_offsets`
  - `map_fds`
  - `packet_offsets`
  - `shared_offsets`
  - `stack_offsets`
  - `shared_region_sizes`
  - `stack_numeric_sizes`
- Helper-effect modeling in `ebpf_domain_t::operator()(const Call&)`.
- Loop/termination analysis via WTO fixpoint plus the `instruction_count` variable.
- Map key/value size and map-type reasoning through `program_info.map_descriptors`.

### What PREVAIL does not already have

- A uniformity lattice.
- Content tracking for non-stack regions such as shared/map-value/context memory.
- GPU helper effects modeled precisely in PREVAIL. The current GPU platform maps helper out-args to `ANYTHING`, not to `PTR_TO_WRITABLE_MEM`, because PREVAIL expects `(ptr, size)` pairs and the GPU helper ABI does not provide them.
- A public API that hands GPU passes the PREVAIL invariant table or BTF summaries.

### Why `SplitDBM` is not the right place to add uniformity

`SplitDBM` is the numeric relation engine. Uniformity is a finite SIMT lattice, not a numeric range/relational property. Reusing PREVAIL does not mean teaching `SplitDBM` about warp uniformity. The right integration layer is:

- a product domain in `crab::ebpf_domain_t`, or
- a sidecar GPU analysis over the same PREVAIL CFG and invariants.

Do not encode uniformity as a fake numeric interval and pretend that is a principled solution.

## A. What PREVAIL Infrastructure Can We Reuse?

### Summary Table

| Gap | PREVAIL reuse | Best integration style | Main blocker |
| --- | --- | --- | --- |
| 1. Memory-sensitive uniformity | Strong reuse for stack types/offsets/cells | Run alongside PREVAIL now; product-domain later | No uniformity lattice, no non-stack content model |
| 2. BTF metadata for uniformity init | Partial reuse already exists in ELF/BTF loader | Plumbing/API change, not Crab change | GPU phases discard BTF/program metadata |
| 3. Side-effect checks on pointee values | Strong reuse for stack pointees and map sizes | Alongside PREVAIL or integrated assertion checks | Current GPU checker only sees reg uniformity |
| 4. Worst-case resource budgets with loop bounds | Strong reuse | Extend PREVAIL internals | Only `instruction_count` exists today |
| 5. Helper metadata instead of hard-coded IDs | Very strong reuse from `gpu_platform.cpp` | Sidecar refactor | GPU passes ignore the metadata already defined |

### Gap 1: Memory-sensitive uniformity

#### What PREVAIL gives us

- `ebpf_domain_t::do_load_stack()` and `do_store_stack()` already know how stack bytes map to semantic cells.
- `array_domain_t` already handles overlapping stack cells and strong/weak invalidation.
- `TypeDomain` already tracks pointer provenance by region.
- `get_type_offset_variable()` already exposes region-relative offsets.

#### What PREVAIL does not directly give us

- No "uniformity" value attached to a register or cell.
- No array domain for shared/map-value/context contents.
- No GPU-specific helper write semantics in PREVAIL core.

#### Recommended integration approach

Near-term:

- Do not keep the current raw-bytecode-only GPU uniformity pass as the primary analysis.
- Move GPU uniformity to the PREVAIL IR/CFG level so it can consume PREVAIL pointer/type/offset facts.
- Implement it as a sidecar analysis first, not as a `SplitDBM` extension.

Long-term:

- If the goal is paper-grade soundness, fold the sidecar into a product domain attached to `crab::ebpf_domain_t`.

#### Can we piggyback on PREVAIL memory-region tracking?

- Stack: yes.
- Packet/shared/context/map-value contents: not as-is.

That means a realistic MVP can become stack-sensitive fairly quickly. A full "memory-sensitive uniformity" claim across all pointees is much harder.

### Gap 2: BTF metadata integration for uniformity initialization

#### What PREVAIL gives us

- `asm_files.cpp` already parses `.BTF` and `.BTF.ext`.
- BTF is already used for:
  - `.maps` parsing into `program_info.map_descriptors`
  - line information in `raw_program.line_info`

#### What is missing

- The GPU verifier API takes only `ebpf_inst*` and `section_name`.
- By the time `uniformity_analysis.cpp` runs, the GPU passes no longer have access to:
  - `raw_program`
  - `program_info`
  - BTF-derived map metadata
  - line info

#### Realistic interpretation

The easiest honest fix is not "parse more BTF." It is:

- stop throwing away the metadata already parsed,
- thread `program_info` and map descriptors into GPU analysis,
- use map key/value sizes and source locations in the GPU checks.

Important limitation:

- BTF is not a magic source of precise local stack-slot layout after optimization.
- If the paper wants to claim uniformity initialization for local variables based on BTF alone, that claim is too strong.
- For optimized compiled BPF, precise source-local-to-stack-slot mapping is a DWARF/debug-info problem and is still brittle.

### Gap 3: Side-effect checks only inspect pointer registers

#### What PREVAIL gives us

- `ValidMapKeyValue` already checks pointed-to ranges using map key/value sizes.
- PREVAIL already knows whether a pointer points into stack vs packet vs shared.
- PREVAIL already tracks stack numeric cells and map metadata.

#### What is missing

- Current GPU SIMT checks do not query pointee contents.
- They only ask "is R2 uniform?" or "is destination register uniform?".

#### Recommended integration approach

- Use PREVAIL provenance and map-size info to derive `(region, offset, width)` for side-effect arguments.
- Then query a GPU uniformity memory domain over that range.
- This is best done alongside PREVAIL invariants or inside a PREVAIL product domain.

Post-processing only the raw register uniformity states is not enough.

### Gap 4: Resource budget is syntactic, not worst-case

#### What PREVAIL gives us

- `instruction_count` variable in the abstract state.
- Loop-aware fixpoint (`wto`, `fwd_analyzer`).
- `get_instruction_count_upper_bound()`.
- Existing termination tests already validate finite vs infinite loops.

#### What is missing

- Only instruction count is modeled.
- No bound variables for:
  - helper calls
  - memory ops
  - map lookups
  - map updates
- The GPU wrapper does not surface PREVAIL stats today.

#### Recommended integration approach

- Extend PREVAIL cost tracking, do not bolt on a second loop-analysis pass.
- Reuse the same fixpoint and add more monotone counters.

This is the cleanest gap to fix with existing PREVAIL infrastructure.

### Gap 5: Uniformity analysis hard-codes helper IDs

#### What PREVAIL / current GPU platform already gives us

`src/gpu/gpu_platform.cpp` already contains:

- per-helper semantic prototype,
- helper argument kinds,
- helper return uniformity.

#### What is missing

- `uniformity_analysis.cpp` ignores it and hard-codes helper IDs.
- `simt_safety_check.cpp` ignores it and hard-codes prohibited helpers.

#### Recommended integration approach

- Refactor GPU passes to query `gpu_platform` metadata.
- Add one richer metadata structure for:
  - return uniformity
  - out-arg write semantics
  - SIMT-prohibited class

This fix is straightforward and should happen before any paper claim about extensibility.

## B. How Complex Is Each Fix?

These estimates are for an honest implementation that would support defensible paper claims. They are not "best-case hack" estimates.

| Gap | Estimated LOC | Difficulty | Can a Codex agent do it end-to-end? | Dependencies |
| --- | ---: | --- | --- | --- |
| 1. Memory-sensitive uniformity | 700-1400 | Hard | Partial yes; full sound version no | 5 helps a lot; 2 helps for sizes/metadata |
| 2. BTF metadata plumbing into GPU analysis | 150-350 | Medium | Yes, if scope is metadata plumbing only | None |
| 2. Full local-variable initialization from metadata | 800+ and still risky | Hard | No, not honestly | None |
| 3. Pointee-sensitive side-effect checks | 300-700 | Medium-Hard | Yes for stack/map-key scope; no for full region coverage | 1 is the main prerequisite |
| 4. Worst-case resource budgets via loop bounds | 250-500 | Medium | Yes | None |
| 5. Helper metadata refactor | 120-250 | Easy-Medium | Yes | None |

### Practical dependency graph

- Gap 5 should be done first. It simplifies gap 1 immediately.
- Gap 4 is mostly independent and should be done early because it is the highest-value low-risk fix.
- Gap 3 depends on gap 1 if we want real pointee checks instead of more pointer-only checks.
- Gap 2 is mostly plumbing. It is useful, but it does not rescue the core soundness problem by itself.

### What a single strong coding agent can likely finish

Likely yes:

- gap 5,
- gap 4,
- gap 2 metadata plumbing,
- a narrowed gap 1/3 implementation limited to stack-resident side-effect arguments and metadata-driven helper out-params.

Unlikely without sustained human design review:

- full region-sensitive uniformity across stack, shared/map-value memory, and arbitrary aliases,
- any claim that BTF fully initializes uniformity of optimized local variables,
- a paper-grade soundness argument for all GPU side effects.

## C. How To Avoid "Faking It?"

### Reviewer tests that would catch a fake implementation

#### Gap 1: Memory-sensitive uniformity

Test 1:

- Program stores `lane_id` into a stack local `key`.
- Program passes uniform pointer `&key` to `bpf_map_update_elem`.
- Expected: reject.
- Fake implementation that only checks pointer uniformity will pass this.

Test 2:

- GPU helper writes varying out-parameter through a stack pointer.
- Later `LDX` reloads the value and branches on it.
- Expected: reject.
- Fake implementation that marks only `R0` varying will pass this.

Test 3:

- Write varying value to stack through one pointer alias, read through another alias.
- Expected: reject.
- Fake slot-name tracking without alias/provenance will miss it.

#### Gap 2: BTF metadata integration

Test 1:

- Use a map whose key/value sizes are only available through ELF/BTF parsing.
- GPU uniformity/safety diagnostics should use the real key size, not a hard-coded `4`.

Test 2:

- Error reports should carry source/line context when BTF line info exists.
- If the paper claims "BTF-integrated diagnostics" but the GPU phase only sees raw bytecode, this fails immediately.

Test 3:

- Build with optimization and changed local layout.
- If the paper claims BTF recovers local stack-slot uniformity, reviewer will try to break that claim with optimized code. Unless there is a precise mapping proof, do not claim it.

#### Gap 3: Pointee-side-effect checks

Test 1:

- Uniform `R2 = &key`, but `key` bytes are varying.
- `bpf_map_update_elem(map, &key, ...)`.
- Expected: reject.

Test 2:

- Uniform pointer to a struct field, but the field value was populated from `thread_idx` or `lane_id`.
- Any side-effect policy that only inspects the pointer register will miss this.

#### Gap 4: Worst-case resource budgets

Test 1:

- Loop body contains one helper call.
- Loop bound is 100.
- Static body count is under budget; worst-case helper count is over budget.
- Expected: reject on helper-call budget.

Test 2:

- Finite but nested loops with known bounds.
- Budget should reflect worst-case total, not only instruction body length.

Test 3:

- Infinite loop or loop with no proven finite bound.
- Expected: reject or report unbounded cost.

#### Gap 5: Metadata-driven helpers

Test 1:

- Add a new GPU helper entry to `gpu_platform.cpp` with `uniformity = VARYING`.
- Uniformity tests should begin respecting it without any change to `uniformity_analysis.cpp`.

Test 2:

- Add a helper marked `simt_prohibited`.
- Safety tests should reject it without editing a hard-coded set in `simt_safety_check.cpp`.

### Soundness invariants that must hold

If these do not hold, the paper should not claim the corresponding property.

1. If a register is labeled `UNIFORM`, all lanes reaching that program point must agree on the full value.
2. If a pointer is labeled `UNIFORM`, lanes must agree on both:
   - region kind
   - byte offset within the region
3. If a memory range is labeled `UNIFORM`, all bytes read by a side effect from that range must be lane-agreeing.
4. Helper metadata must over-approximate all bytes written and all uniformity effects. Missing helper side effects are unsound.
5. Cost counters must be monotone upper bounds under loop joins/widening.
6. Metadata-based initialization must only strengthen the analysis when the mapping is exact. Ambiguous metadata must become `UNKNOWN`, not `UNIFORM`.

### Minimum viable version that is honest

If the full implementation is too large for the paper timeline, the minimum honest version is:

- use metadata-driven helper uniformity and helper prohibition,
- use PREVAIL worst-case instruction bounds,
- add loop-aware cost bounds for helper calls and memory ops,
- make side-effect checks stack-sensitive for stack-resident keys/flags/locals,
- explicitly say non-stack pointee uniformity is not yet modeled generally,
- explicitly say BTF is used for map metadata and diagnostics, not for full local-variable recovery.

That version is still useful and publishable as an engineering step. It just is not the same claim as a fully memory-sensitive GPU verifier.

## D. Concrete Implementation Strategy

## Gap 1: Memory-sensitive uniformity

### Recommendation

Replace the current raw-bytecode GPU uniformity pass with a PREVAIL-aware pass over `InstructionSeq` plus `program_info`, reusing PREVAIL's CFG and pointer facts. Do not try to teach `SplitDBM` about uniformity.

### Files to modify

- `bpftime-verifier/src/gpu/uniformity_analysis.hpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.cpp`
- `bpftime-verifier/src/gpu/gpu_platform.hpp`
- `bpftime-verifier/src/gpu/gpu_platform.cpp`
- `bpftime-verifier/src/gpu/gpu_verifier.cpp`
- `bpftime-verifier/src/bpftime-verifier.cpp`
- `bpftime-verifier/ebpf-verifier/src/crab_verifier.hpp`
- `bpftime-verifier/ebpf-verifier/src/crab_verifier.cpp`

If choosing the stronger product-domain route:

- `bpftime-verifier/ebpf-verifier/src/crab/ebpf_domain.hpp`
- `bpftime-verifier/ebpf-verifier/src/crab/ebpf_domain.cpp`
- add `bpftime-verifier/ebpf-verifier/src/crab/uniformity_domain.hpp`
- add `bpftime-verifier/ebpf-verifier/src/crab/uniformity_domain.cpp`

### Specific functions to change or add

Add or expose:

- `analyze_uniformity(const InstructionSeq&, const program_info&, const PrevailInvariantView&)`
- `get_gpu_helper_effects(int32_t helper_id)`
- `run_ebpf_analysis_with_invariants(...)` or similar internal API that returns CFG plus invariant tables

If integrating deeper:

- `ebpf_domain_t::operator()(const Call&)`
- `ebpf_domain_t::do_load_stack()`
- `ebpf_domain_t::do_store_stack()`
- `ebpf_domain_t::operator()(const Mem&)`

### Pseudocode

```cpp
for each basic block in PREVAIL cfg in fixpoint order:
    in_uniformity = join(pred.out_uniformity)
    for each instruction in block:
        pre_inv = prevail_pre_invariant[instruction.label]
        switch instruction:
        case stack store:
            if base is stack and offset is singleton:
                memory_uniformity.store(stack, offset, width, reg_uniformity[src])
            else:
                memory_uniformity.havoc(stack, possible_range)
        case stack load:
            if base is stack and offset is singleton:
                reg_uniformity[dst] = memory_uniformity.load(stack, offset, width)
            else:
                reg_uniformity[dst] = UNKNOWN
        case helper call:
            effect = gpu_platform.helper_effects(helper_id)
            apply effect to written memory ranges
            reg_uniformity[R0] = effect.return_uniformity
        case pointer arithmetic:
            keep pointer uniform only if region and offset remain uniform
```

### Tests to add

- `gpu_uniformity_stack_pointee_test.cpp`
  - uniform pointer, varying key bytes on stack, must reject
- `gpu_helper_outparam_uniformity_test.cpp`
  - helper writes varying out-param to stack, later branch on reload, must reject
- `gpu_alias_uniformity_test.cpp`
  - varying write through alias, reload through alias, must reject

### Realistic scope boundary

For a near-term paper:

- support stack-resident memory precisely,
- treat non-stack pointee loads as `UNKNOWN` unless a stronger model exists,
- do not claim general shared/map-value/context memory uniformity.

That is conservative and honest.

## Gap 2: BTF metadata integration for uniformity initialization

### Recommendation

Plumb the metadata already parsed by PREVAIL into the GPU phases. Do not claim local-variable recovery from BTF unless you really build and validate that machinery.

### Files to modify

- `bpftime-verifier/ebpf-verifier/src/spec_type_descriptors.hpp`
- `bpftime-verifier/ebpf-verifier/src/asm_files.cpp`
- `bpftime-verifier/src/gpu/gpu_verifier.cpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.hpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.cpp`
- `bpftime-verifier/test/gpu_verifier_e2e_test.cpp`

### Specific changes

- Add a GPU-internal verifier entrypoint that accepts `raw_program` or `program_info`.
- Preserve:
  - map descriptors
  - source line info
  - possibly a compact BTF summary object
- Use map key/value sizes when checking side-effect argument ranges.

### Pseudocode

```cpp
struct GpuAnalysisContext {
    const program_info* info;
    const std::vector<btf_line_info_t>* line_info;
    std::optional<GpuBtfSummary> btf_summary;
};

GpuVerifyResult verify_gpu_program(const raw_program& prog, ...) {
    auto prevail = run_prevail_and_collect(prog);
    auto uniformity = analyze_uniformity(prog_ir, prog.info, prevail, gpu_ctx);
    auto safety = check_simt_safety(prog_ir, prog.info, uniformity, gpu_ctx);
}
```

### Tests to add

- e2e test that safety errors mention real source lines when ELF has `.BTF.ext`
- test that map key sizes are taken from parsed metadata, not hard-coded constants

### Honest limitation

Do not say "BTF initializes stack-slot uniformity" unless you can show an exact compiled-stack mapping. Today that would be an overclaim.

## Gap 3: Side-effect checks must inspect pointee values, not just pointer registers

### Recommendation

Keep the side-effect policy in the GPU layer, but make it query a memory-uniformity range checker built on PREVAIL provenance and map metadata.

### Files to modify

- `bpftime-verifier/src/gpu/simt_safety_check.hpp`
- `bpftime-verifier/src/gpu/simt_safety_check.cpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.hpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.cpp`
- optionally `bpftime-verifier/ebpf-verifier/src/crab/ebpf_domain.cpp` if pushing checks down into PREVAIL assertions

### Specific functions to add

- `require_uniform_memory_range(...)`
- `resolve_pointer_range(...)`
- `map_key_uniformity(...)`

### Pseudocode

```cpp
if helper == BPF_FUNC_map_update_elem:
    map = resolve_map_descriptor(R1)
    key_range = resolve_pointer_range(R2, map.key_size, prevail_pre)
    if !uniformity.is_uniform(key_range):
        reject("map update key bytes are lane-varying")

if helper == BPF_FUNC_map_delete_elem:
    key_range = resolve_pointer_range(R2, map.key_size, prevail_pre)
    if !uniformity.is_uniform(key_range):
        reject(...)
```

For atomics:

```cpp
addr = resolve_pointer_range(dst_reg, access_width, prevail_pre, off)
if !uniformity.is_uniform_pointer(addr):
    reject(...)
```

### Tests to add

- `varying_stack_key_uniform_pointer.bpf.c`
- `varying_delete_key_uniform_pointer.bpf.c`
- `varying_struct_field_side_effect.bpf.c`

### Important policy decision

Decide explicitly whether the uniformity policy constrains:

- only the side-effect target,
- or both target and payload.

Do not quietly enforce one while the paper text implies the other.

## Gap 4: Worst-case resource budgets with loop bounds

### Recommendation

Move budget accounting into PREVAIL's fixpoint, not into a raw instruction scan.

### Files to modify

- `bpftime-verifier/ebpf-verifier/src/config.hpp`
- `bpftime-verifier/ebpf-verifier/src/crab/variable.hpp`
- `bpftime-verifier/ebpf-verifier/src/crab/var_factory.cpp`
- `bpftime-verifier/ebpf-verifier/src/crab/ebpf_domain.hpp`
- `bpftime-verifier/ebpf-verifier/src/crab/ebpf_domain.cpp`
- `bpftime-verifier/ebpf-verifier/src/crab_verifier.cpp`
- `bpftime-verifier/src/bpftime-verifier.cpp`
- `bpftime-verifier/src/gpu/resource_budget.cpp`
- `bpftime-verifier/src/gpu/gpu_verifier.cpp`

### Specific functions to change

- add variables:
  - `helper_count`
  - `memory_op_count`
  - `map_lookup_count`
  - `map_update_count`
- initialize them in `ebpf_domain_t::setup_entry()`
- increment them in:
  - `ebpf_domain_t::operator()(const Call&)`
  - `ebpf_domain_t::operator()(const Mem&)`
  - `ebpf_domain_t::operator()(const LockAdd&)`
- expose upper bounds through stats or a GPU-internal analysis artifact

### Pseudocode

```cpp
void ebpf_domain_t::operator()(const Call& call) {
    add(variable_t::helper_count(), 1);
    if (call.func == BPF_FUNC_map_lookup_elem) add(variable_t::map_lookup_count(), 1);
    if (call.func == BPF_FUNC_map_update_elem) add(variable_t::map_update_count(), 1);
    ...
}

void ebpf_domain_t::operator()(const Mem& m) {
    add(variable_t::memory_op_count(), 1);
    ...
}
```

Then in the GPU budget checker:

```cpp
if prevail_bounds.helper_count > budget.max_helper_calls reject;
if prevail_bounds.memory_op_count > budget.max_memory_ops reject;
```

### Tests to add

- bounded loop with helper in body, worst-case helper budget exceeded
- bounded loop with map update in body, worst-case map-update budget exceeded
- finite loop accepted with exact upper bound under budget
- non-terminating loop rejected as unbounded cost

### Why this is realistic

PREVAIL already solves the hard part: loop-aware fixpoint over the CFG. Adding more monotone counters is a normal extension.

## Gap 5: Uniformity analysis must use `gpu_platform` metadata

### Recommendation

Centralize helper SIMT semantics in `gpu_platform`, then make all GPU passes query that metadata.

### Files to modify

- `bpftime-verifier/src/gpu/gpu_platform.hpp`
- `bpftime-verifier/src/gpu/gpu_platform.cpp`
- `bpftime-verifier/src/gpu/uniformity_analysis.cpp`
- `bpftime-verifier/src/gpu/simt_safety_check.cpp`
- `bpftime-verifier/test/gpu_verifier_test.cpp`

### Specific API additions

Add something like:

```cpp
enum class GpuSimtEffectClass {
    NONE,
    PROHIBITED_SYNC,
};

struct GpuHelperEffect {
    GpuHelperUniformity return_uniformity;
    std::array<GpuHelperArgumentType, 5> args;
    GpuSimtEffectClass effect_class;
};

const GpuHelperEffect& get_gpu_helper_effects(int32_t helper_id);
```

### Pseudocode

```cpp
const auto& proto = get_gpu_helper_semantic_prototype(helper_id);
state.regs[0] = to_uniformity(proto.uniformity);

for each arg i:
    if proto.argument_type[i] == PTR_TO_U64_OUT:
        write_memory_uniformity(reg = i+1, width = 8, value = proto.uniformity);

if proto.effect_class == PROHIBITED_SYNC:
    reject(...)
```

### Tests to add

- add a synthetic GPU helper in test-only metadata and verify the passes follow the metadata automatically
- regression test that no helper ID switches remain in GPU analysis

### Why this matters

Right now the code already has the right metadata in one file and then ignores it everywhere else. That is exactly the kind of discrepancy reviewers treat as paper/implementation drift.

## E. Alternative: Honest Paper Framing

If full implementation is too large for the timeline, the paper should be reframed around what is actually implemented.

### Honest framing that matches the current architecture

Acceptable claim:

- "We extend a standard eBPF verifier with a GPU-specific SIMT policy layer that checks warp-uniform control flow, selected helper-side rules, and bounded resource usage."

Do not claim:

- "full memory-sensitive warp-uniformity verification for arbitrary pointees"
- "BTF-driven uniformity initialization for compiled local variables"
- "worst-case resource budgets" if you still use raw body counts
- "metadata-driven helper reasoning" while helper switches remain hard-coded

### If only a limited near-term fix lands

Then the paper should say something like:

- PREVAIL is used for memory safety, helper validity, and bounded execution.
- The GPU extension adds:
  - metadata-driven helper uniformity,
  - stack-sensitive uniformity for stack-resident side-effect arguments,
  - SIMT policy checks for uniform branches and selected helper prohibitions,
  - loop-aware worst-case instruction/helper/memory budgets.
- The current prototype does not yet reason precisely about all non-stack pointees or all GPU-visible shared-memory effects.

That framing is honest and still technically interesting.

### Suggested claim downgrade if implementation stays close to current code

If the code only reaches the obvious low-risk fixes, the safest phrasing is:

- "GPU-aware policy checks over PREVAIL-verified eBPF"

not

- "a sound SIMT-aware eBPF verifier"

The latter invites reviewer tests on aliasing, pointee values, helper effect coverage, and loop-cost soundness that the current architecture does not yet justify.

## Recommended Execution Order

1. Fix gap 5 first.
2. Fix gap 4 second.
3. Replace raw-bytecode GPU uniformity with a PREVAIL-aware stack-sensitive analysis.
4. Use that analysis to fix gap 3.
5. Plumb BTF/program metadata into GPU diagnostics and size-aware checks.
6. Only then decide whether the paper can honestly claim "memory-sensitive" uniformity beyond stack-resident arguments.

## Bottom Line

PREVAIL already has most of the hard verifier infrastructure needed for a serious GPU verifier:

- stack cells,
- pointer provenance,
- helper-effect modeling,
- map metadata,
- loop-aware fixpoint iteration.

The missing piece is not another ad hoc raw-bytecode pass. The missing piece is to let the GPU policy layer actually consume PREVAIL's state.

If the goal is an honest and defensible paper, the next step is:

- stop duplicating analysis on `ebpf_inst*`,
- move GPU checks onto PREVAIL's CFG and invariants,
- make helper semantics metadata-driven,
- and use PREVAIL's loop analysis for bounded-cost claims.
