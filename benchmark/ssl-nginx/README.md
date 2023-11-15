# nginx test for sslsniff

This test is to show the performance impact of kernel sslsniff and userspace sslsniff. sslsniff is a tool to intercept the ssl handshake and print the packet content of encrypted ssl handshake. The similar approach is very common in modern observability tools and security tools.

This test shoes that:

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
$ ./wrk https://127.0.0.1:4043/index.html -c 100 -d 10
Running 10s test @ https://127.0.0.1:4043/index.html
  2 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     9.18ms   18.95ms 165.75ms   94.73%
    Req/Sec     9.71k     5.05k   32.14k    87.56%
  189498 requests in 10.02s, 49.70MB read
Requests/sec:  18916.15
Transfer/sec:      4.96MB
```

## Test for kernel sslsniff

in one console

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

In another shell

```console
$ ./wrk https://127.0.0.1:4043/index.html -c 100 -d 10
Running 10s test @ https://127.0.0.1:4043/index.html
  2 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    13.59ms   11.30ms 156.13ms   91.27%
    Req/Sec     4.01k     0.95k    5.95k    71.50%
  80242 requests in 10.10s, 21.04MB read
Requests/sec:   7948.46
Transfer/sec:      2.08MB
```

## test for userspace sslsniff

Note: you need to config bpftime to:

1. No locks in hash maps and array maps
2. Using ubpf JIT
3. Using LTO

in one console, start userspace sslsniff

```sh
sudo ~/.bpftime/bpftime load example/sslsniff/sslsniff
```

in another console, restart nginx

```sh
sudo ~/.bpftime/bpftime start nginx -- -c nginx.conf -p benchmark/ssl-nginx
# or sudo LD_PRELOAD=build/runtime/agent/libbpftime-agent.so nginx -c nginx.conf -p benchmark/ssl-nginx
```

in another console, run wrk

```console
$ ./wrk https://127.0.0.1:4043/index.html -c 100 -d 10
Running 10s test @ https://127.0.0.1:4043/index.html
  2 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     6.32ms    6.18ms 164.79ms   97.80%
    Req/Sec     8.41k     1.30k   11.20k    87.37%
  166451 requests in 10.04s, 43.65MB read
Requests/sec:  16580.88
Transfer/sec:      4.35MB
```
