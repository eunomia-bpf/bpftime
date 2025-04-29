# deepflow with userspace uprobe

TODO: more complex for deepflow

with wrk:

```sh
wrk/wrk https://127.0.0.1:446/ -c 500 -t 10 -d 10
```

Test with 4 different types:

1. deepflow with partly userspace uprobe in bpftime
2. deepflow with kernel uprobe, totally running in kernel
3. deepflow without enable uprobe, only kprobe or syscall tracepoints, sockets
4. without deepflow

You should test it with two types of server:

1. Golang server with https enabled, use goroutine to handle requests
2. Golang server with only http enabled, use goroutine to handle requests

## Usage

- Build and run bpftime_daemon. see <https://github.com/eunomia-bpf/bpftime>
- Run the go-server (./go-server/main). If not runnable, build it with `go build main.go`
- Run rust_example from <https://github.com/eunomia-bpf/deepflow/tree/main/build-results> . If unable to run, see <https://github.com/eunomia-bpf/deepflow/blob/main/build.md> for how to build it
- Use attach mode to run bpftime agent. `bpftime attach PID`. PID could be retrived from `ps -aux | grep main`

## Test examples (https)

On my machine:

```console
root@mnfe-pve 
------------- 
OS: Proxmox VE 8.0.4 x86_64 
Host: PowerEdge R720 
Kernel: 6.2.16-19-pve 
Shell: bash 5.2.15 
Terminal: node 
CPU: Intel Xeon E5-2697 v2 (48) @ 3.500GHz 
GPU: NVIDIA Tesla P40 
GPU: AMD ATI Radeon HD 7470/8470 / R5 235/310 OEM 
Memory: 81639MiB / 257870MiB 
```

### HTTPS

These tests were performed using `go-server/main`

#### Without trace
| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
|10 B       |259055.53     |31.13MB       |
|1 KB       |255503.06     |278.27MB      |
|2 KB       |240685.38     |497.17MB      |
|4 KB       |172574.61     |696.72MB      |
|16 KB      |123732.81     |1.90GB        |
|128 KB     |32158.82      |3.93GB        |
|256 KB     |18158.14      |4.44GB        |

#### With kernel uprobe
| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
|10 B       |95356.66      |11.46MB       |
|1 KB       |96107.28      |104.67MB      |
|2 KB       |94280.83      |194.75MB      |
|4 KB       |71658.19      |289.29MB      |
|16 KB      |56206.68      |0.86GB        |
|128 KB     |26142.56      |3.20GB        |
|256 KB     |15227.34      |3.72GB        |
#### With bpftime userspace uprobe (mocked hashmap (by arraymap))

- No userspace lock for shared hashmap
- With LLVM JIT
- Release mode
- ThinLTO
| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
|10 B       |113668.80     |13.66MB       |
|1 KB       |113875.62     |124.02MB      |
|2 KB       |107866.63     |222.82MB      |
|4 KB       |86927.05      |350.94MB      |
|16 KB      |69111.42      |1.06GB        |
|128 KB     |26550.50      |3.25GB        |
|256 KB     |14926.84      |3.65GB        |
### HTTP

These tests were performed using `go-server-http/main`

#### Without trace

| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
| 10 B      |   48174.22   |  5.79MB      |
| 1 KB      |   43417.58   |  47.29MB     |
| 2 KB      |   41130.66   |  84.96MB     |
| 4 KB      |   35208.03   |  142.13MB    |
| 16 KB     |   32904.51   |  518.43MB    |
| 128 KB    |   20155.85   |  2.46GB      |
| 256 KB    |   15352.78   |  3.75GB      |
#### With kernel uprobe

| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
|10 B       |36592.86      |4.40MB        |
|1 KB       |35790.68      |38.98MB       |
|2 KB       |33427.74      |69.05MB       |
|4 KB       |26541.33      |107.15MB      |
|16 KB      |23743.19      |374.09MB      |
|128 KB     |14970.04      |1.83GB        |
|256 KB     |11550.26      |2.82GB        |

#### With bpftime userspace uprobe (mocked hashmap (by arraymap))

- No userspace lock for shared hashmap
- With LLVM JIT
- Release mode
- ThinLTO
| Data Size | Requests/sec | Transfer/sec |
|-----------|--------------|--------------|
|1 KB       |37977.61      |41.36MB       |
|2 KB       |37424.72      |77.31MB       |
|4 KB       |30036.29      |121.26MB      |
|16 KB      |28305.94      |445.98MB      |
|128 KB     |17180.66      |2.10GB        |
|256 KB     |12339.24      |3.01GB        |
