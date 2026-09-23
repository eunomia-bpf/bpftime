## BPF map example

This example tries to create a bpf program which doesn't have to rely on libbpf and which can be tested on systems
that do not have support from libbpf.

To use this program, you would typically need to compile it with an eBPF-capable compiler (like clang with appropriate options),
load it into the kernel using an eBPF loader(`bpftime load` can come to rescue here),
and attach it to the execve system call tracepoint.
The program will then count execve calls per process, which could be read from user space using the exec_count map