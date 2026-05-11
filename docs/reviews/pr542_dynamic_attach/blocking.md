# Blocking Issues (Cross-cutting)

This document lists issues that are high-risk for **security**, **correctness**, or **stability**, and should be addressed before merging PR #542.

## B1. World-writable global shm in `bpftime trace`

- **Location**: `tools/cli/main.cpp:434-445` (`try_relax_global_shm_permissions_for_target`)
- **Problem**: When `bpftime trace` runs under sudo, it attempts to “relax” permissions by doing `chmod(..., 0666)` on `/dev/shm/<BPFTIME_GLOBAL_SHM_NAME>`.
- **Impact**:
  - Multi-user machines: any local user/process can open and write the shared memory segment.
  - Enables cross-tenant tampering/DoS/info-leak; also makes failures non-deterministic.
- **Recommended fix**:
  - Avoid `0666`. Prefer `chown` to target `uid/gid`, then `chmod 0600/0660`.
  - If compatibility with existing root-owned segments is required, implement a safer policy:
    - Detect owner/mode; only adjust when needed.
    - Keep permissions “same uid (or group) only”.
  - Log loudly when adjustment fails (do not silently ignore).

## B2. Late bootstrap “failure becomes permanent” (call_once + swallowed exception + done=true)

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp:1885-1896` (`bootstrap_existing_fatbins_once`)
- **Problem**:
  - `std::call_once` body catches exceptions and does not rethrow.
  - After `call_once`, it unconditionally sets `late_bootstrap_done = true`.
- **Impact**:
  - A transient failure (CUDA context not ready, ELF scan incomplete, PTX extraction fails) can permanently mark bootstrap as “done”.
  - Subsequent refresh/attach may never re-run bootstrap; instrumentation becomes a silent no-op (“no data”) and is hard to diagnose.
- **Recommended fix**:
  - Do not swallow exceptions inside `call_once` (let it throw so `call_once` can retry).
  - Only set `late_bootstrap_done = true` on real success (e.g., after ingest/prefill completes).
  - Consider a tri-state: `not_started / running / succeeded / failed`, so failures remain retryable.

## B3. Agent IPC auth bypass when `SO_PEERCRED` is unavailable

- **Location**: `runtime/agent/agent.cpp:270-279` (IPC accept loop)
- **Problem**: If `getsockopt(SO_PEERCRED)` fails, the code falls through and processes requests anyway.
- **Impact**:
  - In environments where `SO_PEERCRED` is unavailable or fails for abstract sockets, any process that can connect may issue `refresh`/`detach`.
  - This is a control-plane security boundary break.
- **Recommended fix**:
  - Fail closed: if `getsockopt` fails, close the connection and continue.
  - Optional: extend the check to validate `pid`/`uid` more strictly (depending on threat model).

## B4. Potential UB: dereferencing empty `optional` CUDA context in `bpf_attach_ctx`

- **Location**:
  - `runtime/src/attach/bpf_attach_ctx.cpp:138-148` (constructor uses `*cuda::create_cuda_context()`)
  - `runtime/src/attach/bpf_attach_ctx_cuda.cpp:346-354` (`create_cuda_context` may return `std::nullopt`)
- **Problem**: `create_cuda_context()` can return `nullopt` (e.g., missing CUDA shared memory), but the constructor unconditionally dereferences it.
- **Impact**: Undefined behavior / crash during agent init in partial/misaligned environments.
- **Recommended fix**:
  - Change `bpf_attach_ctx` to lazily initialize CUDA context, or store it as `std::optional` and branch safely.
  - Ensure destructor paths are safe when CUDA context never initialized.

## B5. `bpftime trace` loader drop-privileges failures are ignored

- **Location**: `tools/cli/main.cpp:211-217` (child path inside `spawn_command`)
- **Problem**: `setgroups`/`setgid`/`setuid` errors are ignored.
- **Impact**:
  - Loader may continue running as root when it was expected to drop to target uid/gid.
  - Enlarges attack surface and makes shm/permission behavior inconsistent.
- **Recommended fix**:
  - Treat any failure as fatal in the child (log then `_exit(1)`).
  - Consider `setresuid/setresgid`, and ensure capabilities are not retained.

## B6. `bpftime trace` temporary directory lifecycle is not closed

- **Location**: `tools/cli/main.cpp:599-709` (`prepare_cuda_late_ptx_dir`)
- **Problem**:
  - Creates `/tmp/bpftime-late-ptx.XXXXXX` but does not reliably remove it on failure; and has no cleanup policy on success.
- **Impact**: Disk junk accumulation; extracted PTX may contain sensitive code; operational hygiene issues.
- **Recommended fix**:
  - Implement RAII cleanup (remove on all failure paths).
  - For success, either:
    - delete at end of `trace` (after injection succeeds), or
    - delete after agent acknowledges it no longer needs the directory (explicit IPC handshake).

## B7. `bpf_attach_ctx` epoch protocol has TOCTOU window during handler scan

- **Location**: `runtime/src/attach/bpf_attach_ctx.cpp:76-124` (`init_attach_ctx_from_handlers`)
- **Problem**: Reads a stable epoch once, then scans/instantiates handlers for a potentially long time. The session may change mid-scan.
- **Impact**: Mixed instantiation across sessions; hard-to-reproduce crashes or “wrong data”.
- **Recommended fix**:
  - Re-read stable epoch after the scan; if it changed, restart the whole init (or abort and retry).
  - Consider bounding the scan time or introducing a “scan generation id”.

## B8. `read_stable_epoch_seq` masks “writer stuck odd” into an even value

- **Location**: `runtime/src/bpftime_shm_internal.cpp:792-811`
- **Problem**: After max retries, it returns `epoch_seq & ~1`, effectively converting an odd (writer-in-progress) into a fake “stable even”.
- **Impact**:
  - If the writer is stuck/crashed mid-update, readers may accept a bogus stable epoch and proceed.
- **Recommended fix**:
  - On retry exhaustion, return `0` (or an explicit error) and force caller to backoff/retry.
  - Optionally add logging that the epoch could not be stabilized.

