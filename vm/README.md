# bpftime vm: userspace eBPF vm with JIT support

The bpf vm and jit for eBPF usersapce runtime.

you can choose from llvm-jit and a simple-jit/interpreter based on ubpf.

## LLVM jit for eBPF

see [llvm-jit/README.md](llvm-jit/README.md)

## a simple jit modified from ubpf

see [simple-jit/README.md](simple-jit/README.md)

## build

The JIT can be built as a standalone library and integrated into other projects.

In `vm` directory, run:

```sh
make build
```

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

## Roadmap

- [ ] AOT support for LLVM JIT
