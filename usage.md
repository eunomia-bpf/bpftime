# Manual

🚧 It's at an early stage and may contain bugs on more platforms and eBPF programs. We are working on to improve the stability and compatibility. It's not suitable for production use now.

If you find any bugs or suggestions, please feel free to open an issue, thanks!

## Table of Contents

- [Manual](#manual)
  - [Table of Contents](#table-of-contents)
  - [Uprobe and uretprobe](#uprobe-and-uretprobe)
  - [Syscall tracing](#syscall-tracing)
  - [Run with LD\_PRELOAD directly](#run-with-ld_preload-directly)
  - [Configurations for runtime](#configurations-for-runtime)
    - [Run with JIT enabled or disabled](#run-with-jit-enabled-or-disabled)
    - [Run with kernel eBPF and kernel verifier](#run-with-kernel-ebpf-and-kernel-verifier)
    - [Control Log Level](#control-log-level)
    - [Controlling the Log Path](#controlling-the-log-path)
    - [Allow external maps](#allow-external-maps)
    - [Set memory size for shared memory maps](#set-memory-size-for-shared-memory-maps)
  - [Verifier](#verifier)

## Uprobe and uretprobe

With `bpftime`, you can build eBPF applications using familiar tools like clang and libbpf, and execute them in userspace. For instance, the `malloc` eBPF program traces malloc calls using uprobe and aggregates the counts using a hash map.

You can refer to [documents/build-and-test.md](build-and-test.md) for how to build the project.

To get started, you can build and run a libbpf based eBPF program starts with `bpftime` cli:

```console
make -C example/malloc # Build the eBPF program example
bpftime load ./example/malloc/malloc
```

In another shell, Run the target program with eBPF inside:

```console
$ bpftime start ./example/malloc/victim
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
continue malloc...
```

You can also dynamically attach the eBPF program with a running process:

```console
$ ./example/malloc/victim & echo $! # The pid is 101771
[1] 101771
101771
continue malloc...
continue malloc...
```

And attach to it:

```console
$ sudo bpftime attach 101771 # You may need to run make install in root
Inject: "/root/.bpftime/libbpftime-agent.so"
Successfully injected. ID: 1
```

You can see the output from original program:

```console
$ bpftime load ./example/malloc/malloc
...
12:44:35 
        pid=247299      malloc calls: 10
        pid=247322      malloc calls: 10
```

Alternatively, you can also run our sample eBPF program directly in the kernel eBPF, to see the similar output:

```console
$ sudo example/malloc/malloc
15:38:05
        pid=30415       malloc calls: 1079
        pid=30393       malloc calls: 203
        pid=29882       malloc calls: 1076
        pid=34809       malloc calls: 8
```

## Syscall tracing

An example can be found at [examples/opensnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/opensnoop)

```console
$ sudo ~/.bpftime/bpftime load ./example/opensnoop/opensnoop
[2023-10-09 04:36:33.891] [info] manager constructed
[2023-10-09 04:36:33.892] [info] global_shm_open_type 0 for bpftime_maps_shm
[2023-10-09 04:36:33][info][23999] Enabling helper groups ffi, kernel, shm_map by default
PID    COMM              FD ERR PATH
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
```

In another terminal, run the victim program:

```console
$ sudo ~/.bpftime/bpftime start -s example/opensnoop/victim
[2023-10-09 04:38:16.196] [info] Entering new main..
[2023-10-09 04:38:16.197] [info] Using agent /root/.bpftime/libbpftime-agent.so
[2023-10-09 04:38:16.198] [info] Page zero setted up..
[2023-10-09 04:38:16.198] [info] Rewriting executable segments..
[2023-10-09 04:38:19.260] [info] Loading dynamic library..
...
test.txt closed
Opening test.txt
test.txt opened, fd=3
Closing test.txt...
```

## Run with LD_PRELOAD directly

If the command line interface is not enough, you can also run the eBPF program with `LD_PRELOAD` directly.

The command line tool is a wrapper of `LD_PRELOAD` and can work with `ptrace` to inject the runtime shared library into a running target process.

Run the eBPF tool with libbpf:

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

Start the target program to trace:

```sh
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## Configurations for runtime

Some configurations can be set in the environment variables to control the runtime behavior. For the full definition of the environment variables, see [https://github.com/eunomia-bpf/bpftime/blob/master/runtime/include/bpftime_config.hpp](https://github.com/eunomia-bpf/bpftime/blob/master/runtime/include/bpftime_config.hpp).

### Run with JIT enabled or disabled

If the performance is not good enough, you can try to enable JIT. The JIT is enabled by default in new version.

Set `BPFTIME_DISABLE_JIT=true` in the server to disable JIT, for example, when running the server:

```sh
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so BPFTIME_DISABLE_JIT=true example/malloc/malloc
```

The JIT may be disabled in old version. Set `BPFTIME_USE_JIT=true` in the server to enable JIT, for example, when running the server:

```sh
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so BPFTIME_USE_JIT=true example/malloc/malloc
```

The default behavior is using LLVM JIT, you can also use ubpf JIT by compile with LLVM JIT enabled. See [documents/build-and-test.md](build-and-test.md) for more details.

### Run with kernel eBPF and kernel verifier

You can run the eBPF program in userspace with kernel eBPF in two ways. The kernel must have eBPF support enabled, and kernel version should be higher enough to support mmap eBPF map.

- Use `BPFTIME_RUN_WITH_KERNEL` to load the eBPF eBPF application with kernel eBPF loader and kernel verifier. The program will be load into the kernel for verify, but can still run in userspace with bpftime agent.
- Use `BPFTIME_NOT_LOAD_PATTERN` to skip loading the eBPF program into the kernel when the `BPFTIME_RUN_WITH_KERNEL` is set. The pattern is a regular expression to match the program name. This can help skip some userspace only eBPF programs which is not supported by kernel verifier.

1. with the shared library `libbpftime-syscall-server.so`, for example:

```sh
BPFTIME_NOT_LOAD_PATTERN=start_.* BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

2. Using daemon mode, see <https://github.com/eunomia-bpf/bpftime/tree/master/daemon>

### Control Log Level

Set `SPDLOG_LEVEL` to control the log level dynamically, for example, when running the server:

```sh
SPDLOG_LEVEL=debug LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

Available log level include:

- trace
- debug
- info
- warn
- err
- critical
- off

See <https://github.com/gabime/spdlog/blob/v1.x/include/spdlog/cfg/env.h> for more details.

Log can also be controled at compile time by specifying `-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO` in the cmake compile command.

### Controlling the Log Path

You can control the log output path by setting the `BPFTIME_LOG_OUTPUT` environment variable. By default, logs are sent to `~/.bpftime/runtime.log` to avoid polluting the target process. You can override this default behavior by specifying a different log output via the environment variable.

To send logs to `stderr`:

```sh
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

To send logs to a specific file:

```sh
BPFTIME_LOG_OUTPUT=./mylog.txt LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

### Allow external maps

Sometimes you may want to use external maps which bpftime does not support, for example, load a XDP program with a self define map in shared memory, and use own tools to run it.

- Set `BPFTIME_ALLOW_EXTERNAL_MAPS` to allow external(Unsupport) maps load with the bpftime syscall-server library, for example:

```sh
BPFTIME_ALLOW_EXTERNAL_MAPS=true LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so userspace-xdp/xdp_loader
```

### Set memory size for shared memory maps

Sometimes larger maps may need more memory, you can set the memory size for shared memory maps by setting `BPFTIME_SHM_MEMORY_MB` in the server. The size is in MB, for example, when running the server:

```sh
BPFTIME_SHM_MEMORY_MB=1024 LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

## Verifier

Since the primary goal of bpftime is to stay aligned with kernel eBPF, it is recommended to use the kernel's eBPF verifier to ensure program safety.

You can set the `BPFTIME_RUN_WITH_KERNEL` environment variable to allow the program to load into the kernel and be verified by the kernel verifier:

```sh
BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/malloc/malloc
```

If the kernel verifier is not available, you can enable the `ENABLE_EBPF_VERIFIER` option during the bpftime build process to use the `PREVAIL` userspace eBPF verifier:

```sh
cmake -DENABLE_EBPF_VERIFIER=YES -DCMAKE_BUILD_TYPE=Release -S . -B build
```
