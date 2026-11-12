# trace goroutine with uprobe

**Warning**: The offset of goid field is hardcoded. It was only tested on the bundled `go-server-http`. It MAY NOT WORK on other go programs.

The bundled fo program was compiled using go 1.17.0. The executable and source could be found at folder `go-server-http`.

This example traces the state switch of goroutines, and prints the corresponding state, goid, pid and tgid.

## Run the example

Run the tracing prorgam:

```console
example/goroutine# LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./goroutine
TIME     EVENT COMM             PID     PPID    FILENAME/EXIT CODE
```

Run the go program:

```console
example/goroutine# LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so go-server-http/main
Server started!
```

Trigger the go program:

```console
# curl 127.0.0.1:447
Hello,World!
```

## Use bpftime to trace go programs

## Dynamic linkage

Note that since bpftime relies on dynamic library, some go program may not work. This is because the go runtime is statically linked by default, and bpftime cannot trace statically linked programs. In this usecase, the go program use the `net/http` package, which is dynamically linked.

> Go1.1 suports what is called "external" linking, where the system linker is invoked after the go-provided one (6l, 8l, or 5l) to actually perform the final link, and thus supports the full range of features needed to statically link most code properly. As an example, the SQLite package at <http://code.google.com/p/go-sqlite> provides the source code for SQLite and statically links the produced library into your binaries, something that couldn't be done in go 1.0 with the "internal" linker. The "external" linking method should be enabled automatically if you use CGO, although a recent bug has been discovered in the binary distribution of go1.1.1 on darwin (<https://code.google.com/p/go/issues/detail?id=5726>). Note: the "external" linking mode doesn't enforce static linking, it is just better supported; Go will still link to dynamic libraries if any are specified by the cgo arguments, and in the case of the above, there may not be statically linked versions of the libraries it uses.
>
> For the net package, name resolution (net.Lookup*) is normally handled by cgo code because it will then use the same C code that every other program on the system uses and thus produce the same results. If cgo is disabled, a mechanism written in go will be used instead, and it will most likely produce different and fewer results. For os/user, the non-cgo code always returns an error.

Reference:

- <https://github.com/eunomia-bpf/bpftime/issues/221>
- <https://groups.google.com/g/golang-nuts/c/H-NTwhQVp-8>

## LD_PRELOAD and attach
Go has its strange initializaion of ELF executables, it won't invoke __libc_start_main in glibc so injection through LD_PRELOAD won't work. But remote attach `bpftime attach` works.

Console1:
```console
root@mnfe-pve:~/bpftime/example/goroutine# bpftime load ./goroutine
[2024-02-24 19:12:12.559] [info] [syscall_context.hpp:86] manager constructed
[2024-02-24 19:12:12.563] [info] [syscall_server_utils.cpp:24] Initialize syscall server
[2024-02-24 19:12:12][error][3343178] pkey_alloc failed
[2024-02-24 19:12:12][info][3343178] Global shm constructed. shm_open_type 0 for bpftime_maps_shm
[2024-02-24 19:12:12][info][3343178] Enabling helper groups ufunc, kernel, shm_map by default
[2024-02-24 19:12:12][info][3343178] bpftime-syscall-server started
[2024-02-24 19:12:12][info][3343178] Created uprobe/uretprobe perf event handler, module name ./go-server-http/main, offset 38e40
TIME     EVENT COMM             PID     PPID    FILENAME/EXIT CODE
```

Console2:
```console
root@mnfe-pve:~/bpftime/example/goroutine# bpftime attach 3343374
[2024-02-24 19:12:50.606] [info] Injecting to 3343374
[2024-02-24 19:12:50.649] [info] Successfully injected. ID: 1
root@mnfe-pve:~/bpftime/example/goroutine# 
```

Console3:
```console
root@mnfe-pve:~/bpftime/example/goroutine# ./go-server-http/main 
Server started!
[2024-02-24 19:12:50.649] [error] [bpftime_shm_internal.cpp:600] pkey_alloc failed
[2024-02-24 19:12:50.649] [info] [bpftime_shm_internal.cpp:618] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
[2024-02-24 19:12:50.650] [info] [agent.cpp:88] Initializing agent..
[2024-02-24 19:12:50][info][3343464] Initializing llvm
[2024-02-24 19:12:50][info][3343464] Executable path: /root/bpftime/example/goroutine/go-server-http/main
[2024-02-24 19:12:50][info][3343464] Attach successfully
```

Effect:
```console
TIME     EVENT COMM             PID     PPID    FILENAME/EXIT CODE
19:16:35  RUNNABLE  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  DEAD   Goid           0    3344462 3344462
19:16:35  RUNNABLE  Goid           0    3344462 3344462
19:16:35  SYSCALL  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           1    3344462 3344462
19:16:35  WAITING  Goid           1    3344462 3344462
19:16:35  RUNNING  Goid           21   3344466 3344462
19:16:35  SYSCALL  Goid           21   3344466 3344462
19:16:35  RUNNING  Goid           21   3344466 3344462
19:16:35  DEAD   Goid           0    3344466 3344462
19:16:35  RUNNABLE  Goid           0    3344466 3344462
19:16:35  SYSCALL  Goid           21   3344466 3344462
19:16:35  RUNNING  Goid           21   3344466 3344462
19:16:35  WAITING  Goid           21   3344466 3344462
19:16:35  RUNNING  Goid           34   3344464 3344462
19:16:35  RUNNABLE  Goid           21   3344464 3344462
19:16:35  DEAD   Goid           34   3344464 3344462
19:16:35  RUNNING  Goid           21   3344464 3344462
19:16:35  SYSCALL  Goid           21   3344464 3344462
19:16:35  RUNNING  Goid           21   3344464 3344462
19:16:35  WAITING  Goid           21   3344464 3344462
19:16:35  RUNNABLE  Goid           21   3344462 3344462
19:16:35  RUNNING  Goid           21   3344462 3344462
19:16:35  SYSCALL  Goid           21   3344462 3344462
19:16:35  RUNNING  Goid           21   3344462 3344462
19:16:35  SYSCALL  Goid           21   3344462 3344462
19:16:35  RUNNING  Goid           21   3344462 3344462
19:16:35  DEAD   Goid           21   3344462 3344462
```
