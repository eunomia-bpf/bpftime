## Description

<!-- Summarize the change, root cause, and motivation. List any required dependencies. -->

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
- [ ] I have performed a self-review of my own code and my changes generate no new warnings
- [ ] No over-engineering: smallest necessary diff (~50-100 prod lines target where feasible), no speculative abstractions, unused config knobs, or unnecessary new `.so` libraries
- [ ] No over-testing: concise, high-signal regression tests covering the real failure mode without mock/scaffolding bloat
- [ ] Reverses no entry under "Maintainer decisions" in `CLAUDE.md`; a new mechanism (lock, shared-memory layout or attach path, thread, constructor, environment variable, protocol) or a change over ~100 production lines links the issue where a maintainer agreed to it
- [ ] If touching `runtime/` or `attach/`: verified `LD_PRELOAD` safety redlines and hot-path "Three Zeros" (zero libc heap alloc, zero dynamic Boost SHM alloc, zero non-robust cross-process locks; `O_CLOEXEC`; static TLS guard; silent stdio; fail-open)
- [ ] I have filled in the benchmark and performance impact section above with measured numbers
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have made corresponding changes to the documentation where applicable
