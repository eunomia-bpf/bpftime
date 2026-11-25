# `runtime/agent/` guidance

The agent shared object injected into host processes through `LD_PRELOAD`.
Read the root `AGENTS.md` first; its `LD_PRELOAD` Production Safety Redlines
are the binding contract for this directory.

- Constructors must support a fast, zero-allocation early exit before
  touching shared memory, the logger, or Frida: every short-lived child
  utility (`sh`, `sed`, `gcc`, `nvcc`, ...) inherits the preload.
- Never clobber host signal handlers or spawn unsolicited background threads
  in `LD_PRELOAD` mode; preserve `errno` and avoid reentrant `malloc`.
- Stay fork/exec-safe: `O_CLOEXEC`/`SOCK_CLOEXEC` on all internal fds, and no
  lock deadlock in the child after a multi-threaded `fork()`.
- Never sleep/poll (`sleep`/`usleep`/`nanosleep`/`sleep_for`) or take an
  unbounded `flock(LOCK_EX)` in the constructor init path; acquire the SHM
  lifecycle lock non-blockingly and fail open — startup ordering lives in the
  CLI.
- Keep the target's stdout/stderr silent unless console output was explicitly
  selected.
- Build to a low glibc baseline (e.g. 2.17) with `-fvisibility=hidden`, so
  internal C++/Boost/spdlog/Frida symbols never collide with host symbols.
