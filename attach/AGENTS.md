# `attach/` guidance

Event attachment implementations (Frida uprobe, syscall trace, trap, text
segment transformer, NV/GPU). Read the root `AGENTS.md` first; its
`LD_PRELOAD` Production Safety Redlines and Host-process safety invariants
are binding here.

- Uprobe/hook entrypoints use a static (`initial-exec`) TLS recursion guard —
  never dynamic TLS allocation — to stop hook->libc->hook recursion and
  signal-handler reentrancy.
- Hook hot paths allocate nothing: no libc heap allocation and no dynamic
  `segment_manager` shared-memory allocation; preallocate fixed-size
  structures at setup and prefer lock-free atomics or robust shared mutexes.
- Fail open when an attach cannot be serviced: skip the probe or fall back to
  the original behavior; never block, abort, or change the host exit status.
- Hide internal symbols (`-fvisibility=hidden`), and bounds-check resolved
  hook addresses against mapped extents before installing any trampoline.
