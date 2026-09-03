# Trap based uprobe attach backend

A portable implementation of `base_attach_impl` for uprobe, uretprobe,
uprobe-override (filter) and ureplace attaches that does not depend on
frida-gum. It exists for architectures frida does not support: the CMake
option `BPFTIME_ENABLE_TRAP_UPROBE` defaults to ON on riscv64 and OFF
everywhere else, so x86_64 and aarch64 builds keep using frida unchanged.
When the backend is compiled in, `BPFTIME_UPROBE_BACKEND=frida|trap`
selects it at runtime (the agent's environment wins over the loader's
shared config).

## How it works

1. `attach` decodes the first instruction of the target function, saves it,
   and overwrites it with the architecture's breakpoint instruction
   (`int3`, `brk #0`, `ebreak` / `c.ebreak`). The write is a single aligned
   store, so other threads see either the old or the new instruction.
2. The breakpoint raises `SIGTRAP`. The handler looks the faulting pc up in
   an immutable, lock-free probe table, builds a kernel style `pt_regs`
   from the `ucontext_t`, and runs the attached callbacks or eBPF programs
   on the interrupted thread.
3. The original instruction is then either
   - executed out of line: it was copied into an executable slot followed
     by another breakpoint. The handler redirects the pc to the slot, the
     instruction runs, the second trap brings the thread back and the pc is
     set to the instruction after the probe. On x86 a rip-relative
     displacement is adjusted for the slot's address; slots are allocated
     within ±1 GiB of the target for this reason.
   - emulated in software when it is pc-relative or a control transfer
     (`jmp/call/jcc` on x86, `adr/adrp/b/bl/b.cond/cbz/tbz/ldr-literal/blr`
     on aarch64, `auipc/jal/jalr/branches/c.j/c.jr/c.jalr/c.beqz/c.bnez` on
     riscv64).
   Because the instruction is never restored in place, no event is lost when
   many threads hit the probe at once.
4. uretprobes replace the return address (`[rsp]`, `x30`, `ra`) with a
   trampoline that consists of a breakpoint. The original return address is
   kept on a per-thread shadow stack; when the trampoline traps, the return
   callbacks run and the pc is set to the real return address.
5. Filter / replace attaches call `arch::do_return()` on the `ucontext_t`,
   which writes the return value register and returns to the caller without
   executing the function body.

## Host process contract

- The `SIGTRAP` handler chains to whatever handler the host had installed.
  Signals that do not originate from one of our breakpoints are forwarded
  unchanged. If the host installs its own handler later, the next attach
  puts ours back in front and forwards to the new one.
- Nothing in the handler allocates on the first hit of a new thread: the
  shadow stack is `mmap`ed and every thread local it touches uses the
  initial-exec TLS model. Probing `malloc` itself is supported.
- Re-entrancy: if a callback invokes a probed function, the nested hit runs
  the original instruction without callbacks instead of recursing.
- Exceptions thrown by callbacks are caught and logged; the host is never
  terminated by this code.

## Limitations

- Two signal deliveries per hit. Measured on a 32-core riscv64 host
  (Release build): 5.5 µs per uprobe hit and 8.2 µs per uprobe+uretprobe
  pair on one thread; with 8 threads hitting the same probe the aggregate
  throughput barely grows because the kernel serializes signal delivery
  within a process. Use the frida backend where it is available if probe
  overhead matters.
- The first instruction must be decodable and relocatable. Indirect calls
  (`call r/m` on x86), `loop`/`jrcxz`, and functions that already start
  with a breakpoint are rejected with `-ENOTSUP`.
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
| `src/trap_arch_{x86_64,aarch64,riscv64}.cpp` | Decoding, emulation, ucontext access |
| `src/x86_insn_decode.cpp` | Minimal x86-64 instruction length decoder |
| `test/` | Catch2 tests; `bpftime_trap_uprobe_attach_tests` |

The tests run natively and under `qemu-user` for riscv64 and aarch64; see
`cmake/riscv64-toolchain.cmake`.
