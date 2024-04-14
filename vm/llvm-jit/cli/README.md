# bpftime-vm

```console
root@mnfe-pve:~# bpftime-vm
Usage: bpftime-vm [--help] [--version] {build,run}

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 

Subcommands:
  build         Build native ELF(s) from eBPF ELF. Each program in the eBPF ELF will be built into a single native ELF
  run           Run an native eBPF program
```

A CLI program for AOT of llvm-jit.

It can build ebpf ELF into native elf, or execute compiled native ELF. **Helpers and relocations are not supported**

This program might be installed by running `make release-with-llvm-jit` in the project root.
