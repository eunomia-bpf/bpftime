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

Note that since bpftime relies on dynamic library, some go program may not work. This is because the go runtime is statically linked by default, and bpftime cannot trace statically linked programs. In this usecase, the go program use the `net/http` package, which is dynamically linked.

> Go1.1 suports what is called "external" linking, where the system linker is invoked after the go-provided one (6l, 8l, or 5l) to actually perform the final link, and thus supports the full range of features needed to statically link most code properly. As an example, the SQLite package at <http://code.google.com/p/go-sqlite> provides the source code for SQLite and statically links the produced library into your binaries, something that couldn't be done in go 1.0 with the "internal" linker. The "external" linking method should be enabled automatically if you use CGO, although a recent bug has been discovered in the binary distribution of go1.1.1 on darwin (<https://code.google.com/p/go/issues/detail?id=5726>). Note: the "external" linking mode doesn't enforce static linking, it is just better supported; Go will still link to dynamic libraries if any are specified by the cgo arguments, and in the case of the above, there may not be statically linked versions of the libraries it uses.
>
> For the net package, name resolution (net.Lookup*) is normally handled by cgo code because it will then use the same C code that every other program on the system uses and thus produce the same results. If cgo is disabled, a mechanism written in go will be used instead, and it will most likely produce different and fewer results. For os/user, the non-cgo code always returns an error.

Reference:

- <https://github.com/eunomia-bpf/bpftime/issues/221>
- <https://groups.google.com/g/golang-nuts/c/H-NTwhQVp-8>
