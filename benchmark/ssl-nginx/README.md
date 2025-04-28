# nginx test for sslsniff

This test is to show the performance impact of kernel sslsniff and userspace sslsniff. sslsniff is a tool to intercept the ssl handshake and print the packet content of encrypted ssl handshake. The similar approach is very common in modern observability tools and security tools.

## Run with one click script

```sh
cd /path/to/bpftime
python3 benchmark/ssl-nginx/draw_figture.py
```

The result is saved in `size_benchmark_*.txt` and `size_benchmark_*.json`.  You can also check the figture generated.

## Test commands

This test shows that:

1. kernel sslsniff can significantly reduce the performance of nginx, lead to a 2x performance drop.

The test program of sslsniff is from bcc and [bpftime/example](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff). The userspace part modified to not print the packet content out.

## Environment

test env:

```console
$ uname -a
Linux yunwei37server 6.2.0-35-generic #35-Ubuntu SMP PREEMPT_DYNAMIC Tue Oct  3 13:14:56 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
$ nginx -v
nginx version: nginx/1.22.0 (Ubuntu)
$ ./wrk -v
wrk 4.2.0 [epoll] Copyright (C) 2012 Will Glozer
```

## Setup

start nginx server

```sh
nginx -c $(pwd)/nginx.conf -p $(pwd)
```

## Test for no effect

```console
wrk https://127.0.0.1:4043/index.html -c 100 -d 10
```

## Test for kernel sslsniff

in one console

```console
$ make -C example/sslsniff
$ sudo example/sslsniff/sslsniff 
OpenSSL path: /lib/x86_64-linux-gnu/libssl.so.3
GnuTLS path: /lib/x86_64-linux-gnu/libgnutls.so.30
NSS path: /lib/x86_64-linux-gnu/libnspr4.so
FUNC         TIME(s)            COMM             PID     LEN    
lost 194 events on CPU #2
lost 61 events on CPU #3
^CTotal events: 260335 
```

This sslsniff is from bpftime/example/sslsniff/sslsniff. The userspace part modified to not print the packet content out.

In another shell

```console
wrk https://127.0.0.1:4043/index.html -c 100 -d 10  
```

## test for userspace sslsniff

Note: you need to config bpftime to:

1. No locks in hash maps and array maps
2. Using ubpf JIT
3. Using LTO

in one console, start userspace sslsniff

```sh
~/.bpftime/bpftime load example/sslsniff/sslsniff
# or LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/sslsniff/sslsniff
```

in another console, restart nginx

```sh
~/.bpftime/bpftime start nginx -- -c nginx.conf -p benchmark/ssl-nginx
# or LD_PRELOAD=build/runtime/agent/libbpftime-agent.so nginx -c nginx.conf -p benchmark/ssl-nginx
```

in another console, run wrk

```console
$ ./wrk https://127.0.0.1:4043/index.html -c 100 -d 10
```
