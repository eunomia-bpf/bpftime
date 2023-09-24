# opensnoop

Here is a example that demonstrates the usage of userspace syscall trace

## Usage

- Install binutils dev: `sudo apt install binutils-dev`
- Configure & build the project `cmake -S . -B build -G Ninja && cmake --build build --target all --config Debug`. If you encounter errors about `init_disassemble_info`, try `cmake -S . -B build -G Ninja -D USE_NEW_BINUTILS=YES && cmake --build build --target all --config Debug`
- Run `make -C example/opensnoop-printk`
- Run `LD_PRELOAD=./build/runtime/syscall-server/libbpftime-syscall-server.so ./example/opensnoop-printk/opensnoop`
- Run `LD_PRELOAD=./build/runtime/agent-transformer/libbpftime-agent-transformer.so AGENT_SO=./build/runtime/agent/libbpftime-agent.so ./example/opensnoop-printk/victim` in **another** terminal.
- See output. `open pid=XXXXXX, fname=test.txt, comm=victim` marks a successful running.

## Basic benchmark

Measures average time usage when calling open(2)

### Without bpf

```console
Average open time usage 24216.86932ns,  count 176
```

### Use kernel bpf

```console
Average open time usage 29619.08015ns,  count 262
```

### Use userspace syscall trace (ubpf)

```console
Average open time usage 85299.80000ns, count 153
```

### Use userspace syscall trace (llvm-jit)

```console
Average open time usage 25202.18121ns,  count 95
```
