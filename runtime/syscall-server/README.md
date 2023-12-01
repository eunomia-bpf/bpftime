# syscall_server.so

Use as a LD_PRELOAD to intercept bpf syscalls and mock them in userspace, or trace them and make kernel eBPF run with userspace eBPF.

## Run with userspace eBPF

The default behavior is to run with userspace eBPF. Using userspace eBPF means using userspace eBPF maps in shared memory, using userspace eBPF verifier and userspace eBPF runtime.

## run with kernel

Set the environment variable `BPFTIME_RUN_WITH_KERNEL` to `true` to make the kernel eBPF run with userspace eBPF. This means using kernel eBPF maps instead of userspace eBPF maps, and using kernel eBPF verifier instead of userspace eBPF verifier.

```sh
BPFTIME_RUN_WITH_KERNEL=true
```

example:

```sh
SPDLOG_LEVEL=Debug BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```