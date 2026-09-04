# Trap based uprobe attach backend

A riscv64 implementation of `base_attach_impl` for uprobe, uretprobe,
uprobe-override (filter) and ureplace attaches that does not depend on
frida-gum. The CMake option `BPFTIME_ENABLE_TRAP_UPROBE` defaults to ON
on riscv64 (the only supported architecture for this backend).
When the backend is compiled in, `BPFTIME_UPROBE_BACKEND=frida|trap`
selects it at runtime (the agent's environment wins over the loader's
shared config).

## How it works

1. `attach` decodes the first instruction of the target function, saves it,
   and overwrites it with `ebreak` / `c.ebreak`. When the instruction is
   naturally aligned the write is a single atomic store. When a 4-byte
   instruction sits at a 2-byte boundary, a three-phase protocol via
   `c.ebreak` (0x9002) ensures every intermediate state is a valid trap.
2. The breakpoint raises `SIGTRAP`. The handler looks the faulting pc up in
   an immutable, lock-free probe table, builds a kernel style `pt_regs`
   from the `ucontext_t`, and runs the attached callbacks or eBPF programs
   on the interrupted thread.
3. The original instruction is then either
   - executed out of line: it was copied into an executable slot followed
     by another breakpoint. The handler redirects the pc to the slot, the
     instruction runs, the second trap brings the thread back and the pc is
     set to the instruction after the probe.
   - emulated in software when it is pc-relative or a control transfer
     (`auipc/jal/jalr/branches/c.j/c.jr/c.jalr/c.beqz/c.bnez`).
   Because the instruction is never restored in place, no event is lost when
   many threads hit the probe at once.
4. uretprobes replace the return address (`ra`) with a trampoline that
   consists of a breakpoint. The original return address is kept on a
   per-thread shadow stack; when the trampoline traps, the return callbacks
   run and the pc is set to the real return address.
5. Filter / replace attaches call `arch::do_return()` on the `ucontext_t`,
   which writes the return value register and returns to the caller without
   executing the function body.

## Host process contract

- The `SIGTRAP` handler chains to whatever handler the host had installed.
  Signals that do not originate from one of our breakpoints are forwarded
  unchanged. If the host installs its own handler later, the next attach
  puts ours back in front and forwards to the new one.
- The handler path uses only async-signal-safe operations. The per-thread
  uretprobe shadow stack is allocated with `mmap(MAP_ANONYMOUS)` (not
  `malloc`) and every thread-local uses the initial-exec TLS model, so no
  first-access allocation goes through the C library heap. For applications
  that want to eliminate even the `mmap` from the first uretprobe hit,
  `trap_attach_impl::prepare_thread()` can be called once from normal
  (non-signal) context on each thread before probes fire.
- No logging (spdlog / stdio) is performed from inside the signal handler.
  Callback exceptions are caught and silently swallowed; the host is never
  terminated by this code.
- Patching a 4-byte instruction at a 2-byte boundary uses a three-phase
  protocol: (1) write `c.ebreak` into the low half, (2) write the
  intended high half, (3) write the intended low half, with icache flushes
  between phases. Every intermediate state is a trapping compressed
  instruction, so no hart can fetch a torn non-trapping word.
- Re-entrancy: if a callback invokes a probed function, the nested hit runs
  the original instruction without callbacks instead of recursing.

## Limitations

- Two signal deliveries per hit. Measured on a 32-core riscv64 host
  (Release build): 5.5 µs per uprobe hit and 8.2 µs per uprobe+uretprobe
  pair on one thread; with 8 threads hitting the same probe the aggregate
  throughput barely grows because the kernel serializes signal delivery
  within a process.
- The first instruction must be decodable and relocatable. Functions that
  already start with a breakpoint are rejected with `-ENOTSUP`.
- A uretprobe replaces the return address, so unwinding through the probed
  function (C++ exceptions, `backtrace()` from inside it) sees the
  trampoline instead of the caller. This is the same limitation the frida
  backend and kernel uretprobes have.
- Probe sites, out-of-line slots and detached entries are kept alive for
  the lifetime of the process so that a signal handler racing with a
  detach can never touch freed memory.

## Layout

| File | Purpose |
|---|---|
| `include/trap_uprobe_attach_impl.hpp` | Public `trap_attach_impl` class, attach type ids |
| `include/trap_attach_private_data.hpp` | Private data (`addr` or `module:offset`) |
| `include/trap_attach_utils.hpp` | Symbol / module resolution without frida |
| `src/trap_uprobe_attach_impl.cpp` | Engine: probe table, SIGTRAP handler, shadow stack |
| `src/trap_arch.hpp` | Per-architecture interface |
| `src/trap_arch_riscv64.cpp` | Decoding, emulation, ucontext access |
| `test/` | Catch2 tests; `bpftime_trap_uprobe_attach_tests` |

The tests run natively and under `qemu-user` for riscv64; see
`cmake/riscv64-toolchain.cmake`.
