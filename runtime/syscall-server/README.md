# syscall_server.so

Use as a LD_PRELOAD to intercept bpf syscalls and mock them in userspace, or trace them and make kernel eBPF run with userspace eBPF.

## Host-process transparency

The syscall server, agent, and text transformer run inside a process bpftime
does not own. Their exported C and interposition boundaries therefore contain
all exceptions and fail open to the original host operation whenever possible.
Successful fallbacks preserve the original arguments and `errno` behavior.

Initialization is transactional: partial shared-memory, text-permission,
thread, environment, and logger state is rolled back before returning to the
host. Logging is silent by default; stdout or stderr is used only when the user
explicitly selects `BPFTIME_LOG_OUTPUT=console`.

## Run with userspace eBPF

The default behavior is to run with userspace eBPF. Using userspace eBPF means using userspace eBPF maps in shared memory, using userspace eBPF verifier and userspace eBPF runtime.

server

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

client

```sh
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## run with kernel

Set the environment variable `BPFTIME_RUN_WITH_KERNEL` to `true` to make the kernel eBPF run with userspace eBPF. This means using kernel eBPF maps instead of userspace eBPF maps, and using kernel eBPF verifier instead of userspace eBPF verifier.

```sh
BPFTIME_RUN_WITH_KERNEL=true
```

example start tracing:

```sh
SPDLOG_LEVEL=Debug BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

example run target program:

```sh
SPDLOG_LEVEL=Debug LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## skip verification for some programs

```sh
BPFTIME_NOT_LOAD_PATTERN=.*
```
