# `runtime/` guidance

Core userspace eBPF runtime: the shared-memory object registry, BPF maps and
ringbufs, syscall emulation (`syscall-server/`), and the injected agent
(`agent/`). Read the root `AGENTS.md` first; its `LD_PRELOAD` Production
Safety Redlines and Host-process safety invariants are binding here.

- Hot paths (uprobe/syscall callbacks, BPF helpers, ringbuf/map operations,
  constructor early-exit and detach) obey the "Four Zeros": no libc heap
  allocation (and no `static` heap-owning C++ containers with `__cxa_atexit`
  destructors), no dynamic Boost `segment_manager` shared-memory allocation
  (preallocate at setup), no non-robust cross-process locks, and **no
  unapproved active syscalls** (`write`, `read`, `socket`, `connect`,
  `sendmsg`, `recvmsg`, `open`, `mmap`, `eventfd`, etc.; preserve host `errno`
  — only explicit exemptions in `DECISIONS.md` apply).
- Prefer preallocated fixed-size lock-free atomics (L0). Per-CPU maps
  (`PERCPU_ARRAY`, `PERCPU_HASH`), fixed-slot/atomic maps (`ARRAY`,
  `PROG_ARRAY`, `PERF_EVENT_ARRAY`, `BLOOM_FILTER`, `RINGBUF`), and `HASH`
  must never take a global `bpf_map_handler::map_lock` (`should_lock = false`;
  `HASH` and `PERCPU_HASH` share preallocated `bpftime_hash_map` with lock-free
  `lookup` and sharded write stripes). `LPM_TRIE` also keeps
  `should_lock = false`: a preallocated node pool with lock-free `lookup` and
  one recover-and-continue write lock. `HASH_OF_MAPS` stores its 4-byte inner
  map ids in `bpftime_hash_map` and also keeps `should_lock = false`. All cross-process shared-memory lock
  primitives, RAII guards, and blocking-vs-`trylock` dispatch live
  **exclusively** in `runtime/include/robust_shm_mutex.hpp`
  (`bpftime::robust_shm_mutex`, `PTHREAD_PROCESS_SHARED | PTHREAD_MUTEX_ROBUST`):
  forbid defining or calling raw `pthread_mutex_*`, `pthread_spinlock_t`,
  `boost::interprocess::interprocess_*mutex`, custom spinlocks, or bespoke lock
  wrappers outside `runtime/include/robust_shm_mutex.hpp`, and never nest
  private inner locks inside L2 container maps (`LRU_HASH`,
  `QUEUE`, `STACK`, `STACK_TRACE`, where `should_lock = true`
  delegates locking solely to `bpf_map_handler::map_lock` via
  `with_robust_shm_lock`). Adding any new lock, new lock callsite, or lock
  policy change outside the approved L0–L3 inventory in `DECISIONS.md` requires
  an explicit recorded reason and prior maintainer approval.
- Every internal fd (shared memory, eventfd, sockets) is `O_CLOEXEC`/
  `SOCK_CLOEXEC`, and process-local locks stay safe after a multi-threaded
  `fork()`.
- Fail open: unavailable shared memory, a full ringbuf, or an
  `ENOTRECOVERABLE` mutex skips tracing without blocking, aborting, or
  changing the host exit status.
- Shared-memory objects are allocated via `bpftime::shm_construct` /
  `bpftime::shm_find_or_construct` (`runtime/include/bpftime_shm_safe.hpp`),
  enforcing `shm_safe_type<T>` (no `virtual` functions/`vptr`, raw/member
  pointers, C++ references, or process-local STL heap containers/`std::mutex`)
  and `shm_init_ctx` (never store `managed_shared_memory &` in SHM). Plain
  `int` OS file descriptors are likewise forbidden in SHM by review rule; keep
  per-process fds, `mmap` pointers, and CUDA handles in `.bss` tables.
- Init and IPC paths never sleep/poll or take unbounded `flock(LOCK_EX)` or
  untimed socket waits (`SO_SNDTIMEO`/`SO_RCVTIMEO` + `MSG_NOSIGNAL` required):
  use non-blocking acquisition and leave startup ordering to the CLI.
