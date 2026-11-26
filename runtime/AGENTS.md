# `runtime/` guidance

Core userspace eBPF runtime: the shared-memory object registry, BPF maps and
ringbufs, syscall emulation (`syscall-server/`), and the injected agent
(`agent/`). Read the root `AGENTS.md` first; its `LD_PRELOAD` Production
Safety Redlines and Host-process safety invariants are binding here.

- Hot paths (uprobe/syscall callbacks, BPF helpers, ringbuf/map operations,
  constructor early-exit and detach) obey the "Three Zeros": no libc heap
  allocation, no dynamic Boost `segment_manager` shared-memory allocation
  (preallocate at setup), and no non-robust cross-process locks.
- Prefer preallocated fixed-size lock-free atomics. Where mutual exclusion is
  required, use `pthread_mutex_t` with `PTHREAD_PROCESS_SHARED |
  PTHREAD_MUTEX_ROBUST`; never bare `pthread_spinlock_t` or non-robust
  `boost::interprocess::interprocess_mutex`.
- Every internal fd (shared memory, eventfd, sockets) is `O_CLOEXEC`/
  `SOCK_CLOEXEC`, and process-local locks stay safe after a multi-threaded
  `fork()`.
- Fail open: unavailable shared memory, a full ringbuf, or an
  `ENOTRECOVERABLE` mutex skips tracing without blocking, aborting, or
  changing the host exit status.
- Init and IPC paths never sleep/poll or take unbounded `flock(LOCK_EX)` or
  untimed socket waits (`SO_SNDTIMEO`/`SO_RCVTIMEO` + `MSG_NOSIGNAL` required):
  use non-blocking acquisition and leave startup ordering to the CLI.
