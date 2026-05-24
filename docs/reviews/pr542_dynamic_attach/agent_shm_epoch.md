# Agent + Shared Memory Epoch Protocol Review Notes

Files:

- `runtime/src/bpftime_shm_internal.hpp`
- `runtime/src/bpftime_shm_internal.cpp`
- `runtime/src/attach/bpf_attach_ctx.cpp`
- `runtime/include/bpf_attach_ctx.hpp`
- `runtime/agent/agent.cpp`
- `runtime/syscall-server/syscall_server_utils.cpp`

## Summary of intent

PR #542 introduces an `epoch_seq` in shared memory as a “session generation” marker:

- odd: writer mutating/resetting shm state
- even: stable snapshot, `session_id = epoch_seq / 2`

Agent uses it to detect session switches and rebind/detach and re-instantiate handlers.

## Blocking

### A1. IPC auth bypass when `SO_PEERCRED` fails

- **Location**: `runtime/agent/agent.cpp:270-279`
- **Problem**: If `getsockopt(SO_PEERCRED)` fails, request is processed anyway.
- **Fix**: Fail closed (close connection on cred failure). See `blocking.md` (B3).

### A2. `bpf_attach_ctx` CUDA optional dereference can be UB

- **Location**:
  - `runtime/src/attach/bpf_attach_ctx.cpp:138-148`
  - `runtime/src/attach/bpf_attach_ctx_cuda.cpp:346-354`
- **Problem**: `create_cuda_context()` can return `nullopt`, but ctor dereferences it.
- **Fix**: Treat CUDA context as optional and lazy-init; make destructor safe. See `blocking.md` (B4).

### A3. Epoch only coordinates “attach rebind”, not full data-plane safety

- **Location**:
  - Writer: `runtime/src/bpftime_shm_internal.cpp:813-826` (`begin_new_session` calls `reset_server_state()` / `manager->clear_all`)
  - Reader: `runtime/src/attach/bpf_attach_ctx.cpp:76-92` reads epoch at start of init
- **Problem**: Epoch check exists only in (re)initialization path; it does not prevent other code paths (helpers/syscall callbacks) from racing against `clear_all`.
- **Impact**: Protocol may reduce instability but does not guarantee strong consistency during resets.
- **Fix direction**:
  - Document the model explicitly (“epoch protects rebind, not every shm access”).
  - If stronger safety is required: introduce an explicit handshake order (request agent detach -> wait -> clear shm -> bump epoch) and/or data-path epoch validation (higher cost).

### A4. Epoch TOCTOU during handler scan

- **Location**: `runtime/src/attach/bpf_attach_ctx.cpp:76-124`
- **Problem**: Reads epoch once, then scans handlers; session may change mid-scan.
- **Fix**: Re-read epoch after scan; if changed, restart init. See `blocking.md` (B7).

### A5. `read_stable_epoch_seq` returns masked value on retry exhaustion

- **Location**: `runtime/src/bpftime_shm_internal.cpp:792-811`
- **Problem**: On exhaustion, returns `epoch_seq & ~1` (can fabricate stability).
- **Fix**: Return 0 / error; call-site should backoff/retry. See `blocking.md` (B8).

## Should fix

### A6. Session reset call sites need a single source of truth

- **Location**: `runtime/syscall-server/syscall_server_utils.cpp:51-54` calls `begin_new_session()` only during syscall server startup.
- **Problem**: If the intended contract is “each trace session bumps epoch and resets shm”, the call sites should be consistent and explicit (not only at startup).
- **Fix direction**:
  - Decide: session reset is driven by (a) epoch, or (b) explicit agent refresh/detach.
  - Ensure the chosen mechanism is used everywhere the session is restarted.

### A7. Agent IPC responses always return `"ok\n"` even on refresh failure

- **Location**: `runtime/agent/agent.cpp:299-302`
- **Problem**: IPC server ignores `refresh_attach_session` return value and always replies “ok”.
- **Impact**: CLI may assume attach succeeded and proceed with detach/cleanup decisions.
- **Fix**: return a status code or `ok/err` response; have client parse it.

### A8. SIGUSR1 detach can be dropped before pipe is ready

- **Location**:
  - Pipe set in `ensure_detach_worker_started()`
  - Handler uses `detach_pipe_fds[1]` in `sig_handler_sigusr1_detach`
- **Problem**: If SIGUSR1 arrives before pipe init, the request is silently dropped.
- **Fix direction**:
  - Initialize pipe before installing signal handler, or use `sigaction` + safer mechanisms (`signalfd`) where applicable.

### A9. IPC thread teardown uses `detach()` at exit

- **Location**: `runtime/agent/agent.cpp:320-327`
- **Problem**: At exit, thread is detached instead of joined; inconsistent with other shutdown paths.
- **Fix**: Prefer a clear stop flag + close fd + join with bounded wait (or document why detach is safe).

