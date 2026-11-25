# Trap based uprobe attach backend

A riscv64 implementation of `base_attach_impl` for uprobe, uretprobe,
uprobe-override (filter) and ureplace attaches that does not depend on
frida-gum. The CMake option `BPFTIME_ENABLE_TRAP_UPROBE` defaults to ON
on riscv64 (the only supported architecture for this backend).
The backend is selected at compile time: when `BPFTIME_ENABLE_TRAP_UPROBE`
is ON the agent uses the trap backend; otherwise it falls back to frida.

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
- Handler state comes from a statically initialized pool of 1024 slots.
  A thread's first hit claims a slot using lock-free atomics. Idle slots
  are returned once no callback or pending return probe needs them. Only
  small, trivially initialized fields use initial-exec TLS; no dynamic TLS
  resolution, TLS destructor registration, allocation or locking is needed
  on first access. `prepare_thread()` remains optional. Late loading requires
  the loader to have enough static TLS space for these small fields; otherwise
  `dlopen` fails before any breakpoint is installed.
- Runtime-managed BPF links call `prepare_for_signal_execution()` before
  installing a trap. Every helper number and concrete implementation must match the runtime
  list of audited helpers; unaudited helpers return `-ENOTSUP` at admission.
  Currently accepted built-ins are function IP, attach cookie, return override
  and the trap argument/return helpers. Map operations, printing, tail calls,
  dynamic stack collection, AOT objects, custom VMs and MPK execution are not
  admitted. Accepted programs are eagerly JIT-compiled; an unavailable JIT
  also rejects admission. Ordinary execution and other backends retain their
  existing behavior.
- Native callbacks and manually supplied opaque eBPF callbacks must themselves
  obey the async-signal-safe contract: no allocation, locks, logging, throwing,
  or attach/detach operations. The backend cannot inspect arbitrary C++ code.
  Runtime BPF execution rejects an unprepared program even if called manually
  from such a callback. `generate_stack` returns null on this backend.
- Patching a 4-byte instruction at a 2-byte boundary uses a three-phase
  protocol: (1) write `c.ebreak` into the low half, (2) write the
  intended high half, (3) write the intended low half, with icache flushes
  between phases. Every intermediate state is a trapping compressed
  instruction, so no hart can fetch a torn non-trapping word.
- Re-entrancy: if a callback invokes a probed function, the nested hit runs
  the original instruction without callbacks instead of recursing.

## Limitations

- Two signal deliveries per hit. Measured on a 32-core riscv64 host
  (Release build): ~5.7 µs per uprobe hit and ~8.7 µs per
  uprobe+uretprobe pair on one thread; with 8 threads hitting the same
  probe the aggregate throughput barely grows because the kernel
  serializes signal delivery within a process.
- The first instruction must be decodable and relocatable. Functions that
  already start with a breakpoint are rejected with `-ENOTSUP`.
- A uretprobe replaces the return address, so unwinding through the probed
  function (C++ exceptions, `backtrace()` from inside it) sees the
  trampoline instead of the caller. This is the same limitation the frida
  backend and kernel uretprobes have.
- Probe sites, out-of-line slots and detached entries are kept alive for
  the lifetime of the process so that a signal handler racing with a
  detach can never touch freed memory.
- Slot or return-stack exhaustion skips only the affected probe operation and
  resumes the original instruction. A thread exiting with pending return
  probes can retain a pool slot; no per-thread VMA is created in the handler.
- Code patching preserves each page's original permissions and never removes
  EXEC as a fallback for a rejected writable mapping. Restoration failures
  return `-EIO`, even if a retry succeeds. Failed arming rolls back through a
  new permission acquisition; if rollback is denied, the retained trap site
  has no callbacks and still resumes the original instruction. Persistent OS
  refusal can leave permissions degraded and is never reported as success.

## Performance comparison

Per-call latency (ns) across three uprobe backends.  Lower is better.

### Environment

| | x86\_64 (frida / kernel) | riscv64 (trap) |
|---|---|---|
| **CPU** | Xeon Platinum 8259CL 2.50 GHz, 24 cores | SG2042 (C920), 32 cores |
| **Kernel** | 6.8.0 | 6.6 |
| **Build** | Release, ubpf JIT | Release |

### Results (ns/call, avg of 5 runs)

| Probe type | Baseline | Kernel | Frida (userspace) | Trap (userspace) |
|---|---:|---:|---:|---:|
| **uprobe** | 4.9 | 4706 | 1346 | 5654 |
| **uretprobe** | 4.4 | 5444 | 1339 | 8479 |
| **uprobe + uretprobe** | 5.0 | 5782 | 2579 | 8655 |

- All numbers measured 2026-09-05, 5 runs, single thread.
  Frida/kernel: 100k iterations on x86\_64.  Trap: 200k iterations on
  riscv64.
- **Kernel uprobe** crosses into the kernel for every hit; the frida and
  trap backends stay entirely in userspace.
- Frida is ~3.5x faster than kernel uprobe on this workload.
- The trap backend's uretprobe cost (~8.5 µs) dominates uprobe+uretprobe
  because uretprobe itself requires two signal deliveries (entry
  breakpoint + return trampoline breakpoint), same as a standalone
  uprobe+uretprobe pair.

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
`bpftime_trap_late_host` loads `libbpftime_trap_late_module.so` after creating
its worker threads. The module counts handler-reachable allocation, mutex and
dynamic-TLS calls. CTest registers it as `bpftime_trap_late_injection`.

With runtime unit testing and libbpf enabled, run `bpftime_runtime_tests` on
native RISC-V to cover the complete runtime link admission path. The QEMU
workflow runs the trap eBPF integration and direct program admission tests;
the shared-memory link admission test also requires robust futex support
(`set_robust_list`), which qemu-user does not implement. Do not disable
Boost's robust mutexes to make that test pass under emulation.

For native concurrency and signal-safety evidence, run the repository harness
on a riscv64 host:

```console
./.github/script/test-riscv64-native-trap.sh
```

The harness requires a clean tracked worktree, then records the exact commit,
kernel, and CPU topology before running the 2-mod-4 three-phase patch stress
case, allocator/TLS safety cases, and late injection test. It rejects
non-riscv64 and qemu-user execution. Because a full-system riscv64 virtual
machine also reports `riscv64`, attach separate operator provenance when the
result is intended to demonstrate physical-hardware execution.
