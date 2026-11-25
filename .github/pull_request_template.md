## Summary / Root Cause

<!-- What changes and why. Link the issue being fixed and describe the root cause. -->

## Minimal Change & Scope

<!-- Why this is the smallest change that fixes the root cause. If the diff exceeds
roughly 100 lines of production code, justify the size. Call out any new shared
library (.so), shared-memory structure, or configuration knob and explain why it is
strictly necessary. -->

## Verification

<!-- Build and test commands run, and any manual verification performed. -->

## Checklist

- [ ] No over-engineering (no speculative abstractions, unnecessary config knobs, or unrelated refactors)
- [ ] No over-testing (tests are concise and target the real regression without boilerplate bloat)
- [ ] Diff is minimal and no new shared library (.so) was added unless strictly necessary
- [ ] `LD_PRELOAD` / hot-path safety (zero libc heap and dynamic SHM allocation, no non-robust cross-process locks, zero-alloc constructor early-exit, fail-open)
- [ ] Relevant builds and tests pass
