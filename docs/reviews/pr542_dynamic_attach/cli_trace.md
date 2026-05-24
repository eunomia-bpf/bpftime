# `bpftime trace` (CLI) Review Notes

File:

- `tools/cli/main.cpp`

Related:

- `runtime/agent/agent.cpp` (IPC control plane: `refresh/detach/status`)

## Blocking

### C1. World-writable global shm in sudo trace

- **Location**: `tools/cli/main.cpp:434-445`
- **Problem**: `chmod(..., 0666)` on `/dev/shm/<global_shm_name>`.
- **Fix**: See `blocking.md` (B1).

### C2. Drop-privileges failures are ignored

- **Location**: `tools/cli/main.cpp:211-217` (child process path)
- **Problem**: `setgroups/setgid/setuid` results are ignored.
- **Fix**: See `blocking.md` (B5).

### C3. Temporary PTX directory lifecycle is not closed

- **Location**: `tools/cli/main.cpp:599-709`
- **Problem**: `/tmp/bpftime-late-ptx.*` is not reliably removed.
- **Fix**: See `blocking.md` (B6).

### C4. Ctrl+C teardown uses fixed sleep, no reliable “detach completed” sync point

- **Location**:
  - `tools/cli/main.cpp:1127-1141` (Ctrl+C path)
  - `tools/cli/main.cpp:1149-1157` (final detach)
- **Problem**: After sending detach (IPC + SIGUSR1), CLI waits a fixed `usleep(200ms)` and then stops loader.
- **Impact**: On slow systems, detach may not complete; loader teardown may still race with CUDA IPC usage.
- **Fix direction**:
  - Make IPC `detach` blocking and return “done” only after `destroy_all_attach_links()` finishes (or explicit timeout).
  - Alternatively poll `status` until state transitions, then stop loader.
  - Make timeout configurable and warn loudly on timeout.

## Should fix

### C5. `prepare_cuda_late_ptx_dir` uses `/bin/sh -c` — better to argv-ize

- **Location**: `tools/cli/main.cpp:626-672`
- **Current state**: Inputs are single-quoted; injection risk is low but non-zero and increases audit complexity.
- **Recommendation**:
  - Execute `cuobjdump` directly via `posix_spawn` with argv, and use file actions or a small `fork+chdir+execve` wrapper to avoid shell parsing.

### C6. IPC refresh success is not verified

- **Location**: `tools/cli/main.cpp:1107-1111` sets `agent_attached=true` if connect succeeds.
- **Related**: Agent replies `ok\n` even when refresh fails (`runtime/agent/agent.cpp:299-302`).
- **Recommendation**:
  - Return `ok/err` plus details from agent; parse response before deciding `agent_attached`.

### C7. Loader environment is inherited almost completely

- **Location**: `tools/cli/main.cpp` (spawn env construction around `spawn_command`)
- **Problem**: In sudo scenarios, inherited environment may carry sensitive vars; combined with privilege-drop issues this is risky.
- **Recommendation**:
  - Build a minimal env allowlist for the loader (PATH + required CUDA/LLVM vars).
  - Explicitly clear `LD_*` and other high-risk variables.

## Optional

### C8. Unused state / clarity improvements

- **Location**: `tools/cli/main.cpp` (e.g., `g_trace_loader_pid` written but not used)
- **Recommendation**: Remove dead state or use it consistently to avoid confusing future maintainers.

