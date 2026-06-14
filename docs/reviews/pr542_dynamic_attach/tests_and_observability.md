# Tests & Observability — Recommended Validation Plan

This is a “minimal but effective” validation plan for PR #542, focusing on the highest-risk areas (security boundaries, lifecycle correctness, session resets, and CUDA late attach behavior).

## 1) Security boundary checks

### T1. Agent IPC auth should fail closed

- **Target**: `runtime/agent/agent.cpp` IPC server
- **Test idea**:
  - Simulate/force `getsockopt(SO_PEERCRED)` failure and ensure the connection is rejected (no `refresh/detach` executed).
  - Validate that uid mismatch is rejected.

### T2. Shared memory permissions under sudo trace

- **Target**: `tools/cli/main.cpp` `try_relax_global_shm_permissions_for_target()`
- **Test idea**:
  - Run `bpftime trace` as root against a target owned by another user.
  - Assert shm permissions are not world-writable; verify target uid can still use it.

## 2) Session / epoch correctness

### T3. Epoch read stability under writer pressure

- **Target**: `runtime/src/bpftime_shm_internal.cpp` `read_stable_epoch_seq()`
- **Test idea**:
  - In one thread/process: continuously bump epoch (odd->even).
  - In another: read stable epoch and ensure it never returns fabricated stable values when writer is stuck odd (should return 0/error/backoff).

### T4. TOCTOU: handler scan should not mix sessions

- **Target**: `runtime/src/attach/bpf_attach_ctx.cpp` `init_attach_ctx_from_handlers()`
- **Test idea**:
  - Introduce a forced session bump mid-scan.
  - Ensure init detects epoch change and retries/aborts instead of partially instantiating mixed handlers.

## 3) CUDA late attach behavior

### T5. Late bootstrap retryability

- **Target**: `attach/nv_attach_impl/nv_attach_impl.cpp` late bootstrap state machine
- **Test idea**:
  - Make bootstrap fail once (e.g., force missing `cuobjdump` / invalid PTX dir), then fix environment and trigger refresh.
  - Verify bootstrap re-runs and instrumentation resumes (no permanent “done=true but empty cache”).

### T6. Duplicate kernel name collision handling

- **Target**: prefill `kernel_name -> CUfunction` mapping
- **Test idea**:
  - Load two modules with same kernel name.
  - Ensure either collision is detected and disabled, or mapping is made unambiguous by module identity.

## 4) CLI teardown safety

### T7. Ctrl+C ordering and detach completion

- **Target**: `tools/cli/main.cpp` Ctrl+C path
- **Test idea**:
  - Under stress (long-running kernels / slow detach), press Ctrl+C repeatedly.
  - Verify loader is not killed before detach completes (or CLI warns and uses a bounded, visible timeout strategy).

## 5) Hygiene

### T8. Temporary directories are cleaned up

- **Target**: `tools/cli/main.cpp` `prepare_cuda_late_ptx_dir()` and nv_attach temp dirs
- **Test idea**:
  - Run trace success and failure paths; confirm `/tmp/bpftime-late-ptx.*` and other work dirs are removed (unless explicit debug “keep”).

