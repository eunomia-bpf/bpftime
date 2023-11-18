# nginx test for sslsniff

This test is to show the performance impact of kernel sslsniff and userspace sslsniff. sslsniff is a tool to intercept the ssl handshake and print the packet content of encrypted ssl handshake. The similar approach is very common in modern observability tools and security tools.

This test shoes that:

1. kernel sslsniff can significantly reduce the performance of nginx, lead to a 2x performance drop.

The test program of sslsniff is from bcc and [bpftime/example/sslsniff](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff). The userspace part modified to not print the packet content out.

## Environment

test env:

```console
$ uname -a
Linux yunwei37server 6.2.0-35-generic #35-Ubuntu SMP PREEMPT_DYNAMIC Tue Oct  3 13:14:56 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
$ nginx -v
nginx version: nginx/1.22.0 (Ubuntu)
$ ./wrk -v
wrk 4.2.0 [epoll] Copyright (C) 2012 Will Glozer
$ lshw
12th Gen Intel(R) Core(TM) i9-12900H
$ nproc
8
```

Run with 4 threads and 512 connections

## Setup

start nginx server

```sh
nginx -c $(pwd)/nginx.conf -p $(pwd)
```

## Test for no effect

You should test each for 10 seconds, and record the result in test-log.txt. repeated 3 times.

```console
$ make test-log.txt
wrk/wrk https://127.0.0.1:4043/index.html -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example1k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example2k.txt -c 512 -t 4 -d 10 >> test-log.txt
...
```

| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
| 30 B      |              |              |
| 1 KB      |              |              |
| 4 KB      |              |              |
| 16 KB     |              |              |
| 64 KB     |              |              |
| 256 KB    |              |              |

## Test for kernel sslsniff

In one console, it will hook nginx:

```console
$ sudo ./sslsniff 
OpenSSL path: /lib/x86_64-linux-gnu/libssl.so.3
GnuTLS path: /lib/x86_64-linux-gnu/libgnutls.so.30
NSS path: /lib/x86_64-linux-gnu/libnspr4.so
FUNC         TIME(s)            COMM             PID     LEN    
lost 194 events on CPU #2
lost 61 events on CPU #3
^CTotal events: 260335 
```

This sslsniff is from bpftime/example/sslsniff/sslsniff. The userspace part modified to not print the packet content out.

| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
| 30 B      |              |              |
| 1 KB      |              |              |
| 4 KB      |              |              |
| 16 KB     |              |              |
| 64 KB     |              |              |
| 256 KB    |              |              |

## test for userspace sslsniff

Note: you need to config bpftime to:

1. No locks in hash maps and array maps
2. Using ubpf JIT
3. Using LTO

in one console, start userspace sslsniff

```sh
sudo BPFTIME_USE_JIT=true LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/sslsniff/sslsniff
```

in another console, restart nginx

```sh
sudo BPFTIME_USE_JIT=true LD_PRELOAD=build/runtime/agent/libbpftime-agent.so nginx -c nginx.conf -p benchmark/ssl-nginx
# or sudo LD_PRELOAD=build/runtime/agent/libbpftime-agent.so nginx -c nginx.conf -p benchmark/ssl-nginx
```

| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
| 30 B      |              |              |
| 1 KB      |              |              |
| 4 KB      |              |              |
| 16 KB     |              |              |
| 64 KB     |              |              |
| 256 KB    |              |              |
