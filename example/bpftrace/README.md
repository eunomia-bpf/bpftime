# bpftrace example: run bpftrace in userspace

bpftrace is a high-level tracing language for Linux enhanced Berkeley Packet Filter (eBPF) available in recent Linux kernels (4.x). bpftrace uses LLVM as a backend to compile scripts to BPF-bytecode and makes use of BCC for interacting with the Linux BPF system, as well as existing Linux tracing capabilities: kernel dynamic tracing (kprobes), user-level dynamic tracing (uprobes), and tracepoints. The language is inspired by awk and C, and predecessor tracers such as DTrace and SystemTap.

Repo: <https://github.com/iovisor/bpftrace>

However, running bpftrace requires root privileges and eBPF availability in the kernel. With bpftime, you can run bpftrace in userspace, without kernel eBPF support.

Note: The operating system of the .bt file of our test warehouse instance is Ubuntu 22.04. Under this version of the operating system, enter the command `apt install bpftrace` to get bpftrace version 0.9.4 .

## uprobe example

This is an example, you can first run a bpftrace command in userspace with uprobe:

```console
$ sudo ~/.bpftime/bpftime load bpftrace -e 'uretprobe:/bin/bash:readline { printf("%-6d %s\n", pid, str(retval)); }'
[2023-10-27 19:00:32][info][13368] bpftime-syscall-server started
Attaching 1 probe...
13615  
13615  clear
13615  clear
13615  cat
```

And then start a bash process, you can see the output from bpftrace.

```console
sudo ~/.bpftime/bpftime start /bin/bash
```

We've test the bpftrace with [this commit](https://github.com/iovisor/bpftrace/commit/75aca47dd8e1d642ff31c9d3ce330e0c616e5b96). The older version of bpftrace may introduce some bugs.

## Syscall tracing example

```console
$ sudo SPDLOG_LEVEL=error ~/.bpftime/bpftime load bpftrace -e 'tracepoint:syscalls:sys_enter_openat { printf("%s %s\n", comm, str(args->filename)); }'
[2023-10-27 19:17:34.099] [info] manager constructed
[2023-10-27 19:17:34.289] [info] Initialize syscall server
Attaching 1 probe...
cat /usr/lib/locale/locale-archive
cat build/install_manifest.txt
```

And then:

```console
$ sudo SPDLOG_LEVEL=error ~/.bpftime/bpftime start -s cat build/install_manifest.txt
[2023-10-27 19:17:41.677] [info] Entering bpftime agent
[2023-10-27 19:17:41.986] [info] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
/root/.bpftime/bpftime_daemon
/root/.bpftime/bpftimetool
/root/.bpftime/libbpftime-agent.so
/root/.bpftime/libbpftime-agent-transformer.so
/root/.bpftime/libbpftime-syscall-server.so
```

You can also trying bpftrace in userspace with the following one-liner and tools in this dir.

## One-Liners

Here are some possible one-liners to try out with bpftrace, you can run with userspace bpftime. They demonstrate different capabilities:

```sh
# Files opened by process
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { printf("%s %s\n", comm, str(args->filename)); }'

# Syscall count by program
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Read bytes by process:
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret/ { @[comm] = sum(args->ret); }'

# Read size distribution by process:
bpftrace -e 'tracepoint:syscalls:sys_exit_read { @[comm] = hist(args->ret); }'

# Show per-second syscall rates:
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @ = count(); } interval:s:1 { print(@); clear(@); }'
```

More powerful scripts can easily be constructed. See the other tools for examples. Note: some scripts may not work as expected due to the lack of helpers, we are working on it.
