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

## Checklist

- [ ] My code follows the style guidelines and `AGENTS.md` / `CLAUDE.md` of this project
- [ ] I have performed a self-review of my own code and my changes generate no new warnings
- [ ] No over-engineering: smallest necessary diff (~50-100 prod lines target where feasible), no speculative abstractions, unused config knobs, or unnecessary new `.so` libraries
- [ ] No over-testing: concise, high-signal regression tests covering the real failure mode without mock/scaffolding bloat
- [ ] If touching `runtime/` or `attach/`: verified `LD_PRELOAD` safety redlines and hot-path "Three Zeros" (zero libc heap alloc, zero dynamic Boost SHM alloc, zero non-robust cross-process locks; `O_CLOEXEC`; static TLS guard; silent stdio; fail-open)
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have made corresponding changes to the documentation where applicable
