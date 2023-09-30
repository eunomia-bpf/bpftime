# opensnoop

Here is a example that demonstrates the usage of userspace syscall trace

## Usage

- Install binutils dev: `sudo apt install binutils-dev`
- Configure & build the project `cmake -S . -B build -G Ninja && cmake --build build --target all --config Debug`. If you encounter errors about `init_disassemble_info`, try `cmake -S . -B build -G Ninja -D USE_NEW_BINUTILS=YES && cmake --build build --target all --config Debug`

## build

```sh
make -C benchmark/syscall/
```

## run

```sh
sudo LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so  benchmark/syscall/syscall
```

in another shell, run the target program with eBPF inside:

```sh
sudo LD_PRELOAD=build/runtime/agent/libbpftime-agent.so benchmark/syscall/victim
```

## baseline

Average time usage 938.53511ns,  count 1000000

## userspace syscall

Average time usage 492.04251ns,  count 1000000

## kernel tracepoint

Average time usage 423.72835ns,  count 1000000
