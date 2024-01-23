# bpftime vm: userspace eBPF vm with JIT/AOT support

The bpf vm and JIT/AOT for eBPF usersapce runtime.

you can choose from llvm-jit and a simple-jit/interpreter based on ubpf.
The JIT can be built as a standalone library and integrated into other projects.
You can also try the cli tool to compile and run AOT eBPF programs.

Build the vm only:

```sh
make build-llvm # build llvm backend
make build-ubpf # build ubpf backend
```

## LLVM JIT/AOT for eBPF

see [llvm-jit/README.md](llvm-jit/README.md).

You can also build the llvm JIT/AOT for eBPF as a standalone library in it's own directory:

```sh
sudo apt install llvm-15-dev
cd llvm-jit
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -j
```

You can see the cli for how to use [AOT compile](cli/README.md).

## a simple jit modified from ubpf

see [simple-jit/README.md](simple-jit/README.md)

> Note: we will use ubpf to replace the simple-jit in the future.

## Example Usage

See [example/main.c](example/main.c) for how to use it.

## cli

A tool for loading and running eBPF programs.

```console
$ bpftime-cli
Usage: build/vm/cli/bpftime-cli <path to ebpf instructions> [path to memory for the ebpf program]
```

## benchmark

see [github.com/eunomia-bpf/bpf-benchmark](https://github.com/eunomia-bpf/bpf-benchmark) for how we evaluate and details.
