## Description

**Problem statement**:

<!-- State the exact reproduced bug, failure symptom, or missing capability this PR solves. Cite the failing command, test, workload, or issue. -->

**Necessity & why existing primitives are insufficient**:

<!-- Explain the root cause and why existing OS/kernel/libc/Boost/bpftime primitives or configuration cannot solve it directly. Do not hand-roll replacements for standard primitives without proving they fail here. -->

**Scope & diff size (`prod +X/-Y`, `test +A/-B`, `doc +C/-D`)**:

<!-- Keep to one independent root cause or cohesive capability (~50-100 prod lines target where feasible). Confirm zero unrelated refactors, speculative abstractions, or extra bundled changes. -->

Fixes # (issue)

## Type of change

<!-- Delete options that are not relevant. -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Refactor / cleanup (simplifies or reduces code without changing behavior)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?

<!-- Describe the build, unit tests, and integration tests you ran to verify your changes. -->

**Test Configuration**:

- OS / Kernel:
- Arch:
- Toolchain / Compiler:
- Build options (e.g. LLVM / CUDA / Frida / syscall-server):

## Benchmark and performance impact

<!-- Required for every pull request. -->

**Performance impact analysis**:

<!-- Which hot paths does this change touch (uprobe/syscall callbacks, BPF helpers, map, ring buffer and perf event operations, attach/detach, agent startup under LD_PRELOAD)? What does it add or remove on them: instructions, atomics, locks, allocations, syscalls? If it touches none of them, name the paths you checked and why they are unaffected. -->

**Benchmark** (master vs. this PR, same machine and build options; include a run with several threads or processes when the change touches a lock or shared state):

| Benchmark | master | this PR | change |
| --- | --- | --- | --- |
|  |  |  |  |

- Command(s):
- Machine (CPU, cores) and runs per measurement:

## Checklist

- [ ] My code follows the style guidelines and `AGENTS.md` / `CLAUDE.md` of this project
- [ ] Single responsibility (one PR, one thing): addresses one independent root cause or cohesive capability without bundling orthogonal changes ("A and B")
- [ ] Necessity proven: explicitly stated the reproduced problem, why existing OS/library/runtime primitives cannot solve it directly, and exact `prod / test / doc` line deltas with zero extra baggage
- [ ] I have performed a self-review of my own code and my changes generate no new warnings
- [ ] No over-engineering: smallest necessary diff (~50-100 prod lines target where feasible), no speculative abstractions, unrequested teardown/GC state machines without a production caller, test-only IPC/control-plane backdoors, parallel rewrites of already-idempotent functions, unused config knobs, or unnecessary new `.so` libraries
- [ ] No over-testing: concise, high-signal regression tests covering the real failure mode without mock/scaffolding bloat
- [ ] Reverses no entry in `DECISIONS.md`; adds no cross-process lock primitive or wrapper outside `runtime/include/robust_shm_mutex.hpp`; any new lock, new lock callsite, lock policy change, new mechanism (shared-memory layout or attach path, thread, constructor, environment variable, protocol), or change over ~100 production lines records its reason and links the issue or pull-request comment where a maintainer agreed to it
- [ ] If touching `runtime/` or `attach/`: verified `LD_PRELOAD` safety redlines and hot-path "Four Zeros" (zero libc heap alloc / no `__cxa_atexit` static heap containers, zero dynamic Boost SHM alloc, zero non-robust cross-process locks, zero unapproved active syscalls while preserving host `errno`; `O_CLOEXEC`; static TLS guard; silent stdio; fail-open)
- [ ] I have filled in the benchmark and performance impact section above with measured numbers
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have made corresponding changes to the documentation where applicable
