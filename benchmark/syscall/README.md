# opensnoop

Here is a example that demonstrates the usage of userspace syscall trace

## Usage

- Install binutils dev: `sudo apt install binutils-dev`
- Configure & build the project `cmake -S . -B build -G Ninja && cmake --build build --target all --config Debug`.

## build

```sh
make -C benchmark/syscall/
```

## run

Start the eBPF program

```sh
sudo LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so  benchmark/syscall/syscall
```

in another shell, run the target program with eBPF inside:

```sh
sudo ~/.bpftime/bpftime start -s benchmark/syscall/victim
```

## baseline

Average time usage 213.62178ns,  count 1000000

## userspace syscall

Average time usage 446.19869ns,  count 1000000

## kernel tracepoint

Average time usage 365.44980ns,  count 1000000
