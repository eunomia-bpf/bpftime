# PR #542 Review Notes — Dynamic Attach (CUDA)

This folder captures a structured code review of GitHub PR **eunomia-bpf/bpftime#542** (“Support dynamic attach”).

## Scope

- Base: `origin/master`
- Head: `dev/dynamic_attach`
- Diff size: 36 files changed, +5114 / -688, 21 commits
- Focus: CUDA late attach bootstrap, launch routing fallback, shared-memory session/epoch protocol, single-agent control plane (IPC/refresh/detach), CLI `bpftime trace`, GPU CI and examples.

## Severity

- **Blocking**: Should be fixed before merge (security/correctness/stability).
- **Should fix**: Strongly recommended; likely to cause flakes, operational pain, or hard-to-debug gaps.
- **Optional**: Improvements and maintainability polish.

## Documents

- `blocking.md`
  - Cross-cutting “must fix” items (security boundaries, correctness, irrecoverable states).
- `nv_attach_impl.md`
  - CUDA attach implementation review (`attach/nv_attach_impl/**`).
- `agent_shm_epoch.md`
  - Agent control plane + shared-memory epoch protocol (`runtime/agent/**`, `runtime/src/bpftime_shm_internal.*`, `runtime/src/attach/**`).
- `cli_trace.md`
  - `bpftime trace` flow and process orchestration (`tools/cli/main.cpp`).
- `ci_workflows.md`
  - GitHub Actions changes and risks (`.github/workflows/**`).
- `gpu_maps.md`
  - GPU map-side changes and correctness notes (`runtime/src/bpf_map/gpu/**`).
- `tests_and_observability.md`
  - Test gaps and recommended “minimal but effective” validation plan.

## Notes

- File/line references are based on the local checkout at the time these notes were written; they may drift as the branch evolves.
- These notes are intended to be actionable (each item includes concrete location + recommended fix direction).

