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
- [ ] Relevant builds and tests pass
