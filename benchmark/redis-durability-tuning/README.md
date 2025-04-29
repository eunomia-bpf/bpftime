# Redis Durability Tuning with BPFtime

This project explores using BPFtime to enhance Redis durability while minimizing performance impact.

## Background

Redis, a key-value store, offers durability through a write-ahead log (called an Append-Only File (AOF)). By default, Redis provides three durability configurations that tradeoff durability and performance overhead:

- **no AOF**: Does not provide durability
- **everysec**: Ensures that writes are durable every second
- **alwayson**: Ensures that every write is immediately durable

The durability gap between alwayson and everysec is substantial: a crash under everysec can lead to the loss of tens of thousands of updates. Unfortunately, alwayson also reduces throughput by a factor of six compared to everysec.

## Approach

Redis can use BPFtime to provide customizable durability through extensions. Since Redis AOF is not currently designed for extensibility, we added new functions to the source code to support extensibility. We defined three new functions and called them at the top of Redis's functions for write, fsync, and fdatasync.

Then, we used BPFtime annotations to identify these new functions as extension entries. Altogether, this change required about 20 lines of code. With the updated Redis, the system administrator can explore custom durability policies.

## Implementations

We've implemented several different approaches to improve Redis durability:

### 1. Delayed fsync (delayed-fsync)

This extension extends the behavior of fdatasync so that it waits for the previous call to fdatasync, which ensures that the system loses at most 2 updates.

### 2. Fast-path optimization (fsync-fast-notify)

This approach extends the kernel to expose a shared variable to userspace that tracks the number of completed fdatasync operations on each of a process' open files. The fdatasync extension reads from the shared variable and only executes system calls when the previous fdatasync has not completed.

### 3. BPF Sync Kernel (bpf-sync-kernel)

An additional kernel-level approach for optimizing sync operations.

## Prerequisites

Before building and running the benchmark, make sure you have the following installed:

- GCC/G++ compiler
- Make
- Git
- Python (for running the benchmark script)
- Clang (for BPF compilation)
- LibELF development files (`libelf-dev` or equivalent)

## Building and Running

### Building Redis

The benchmark uses a modified version of Redis that includes support for BPFtime extensions. To build Redis:

```bash
make redis
```

This will:
1. Clone the Redis repository from https://github.com/eunomia-bpf/redis
2. Build Redis from source

### Building BPF Extensions

Each of the BPF extension implementations has its own Makefile. You can build all of them by running:

```bash
make build
```

This command builds the `batch_process` implementation. To build all implementations individually:

#### Building Individual Implementations

**Batch Process Implementation:**
```bash
cd batch_process
make
```

**Delayed-Fsync Implementation:**
```bash
cd delayed-fsync
make
```

**Fast-Notify Implementation:**
```bash
cd fsync-fast-notify
make
```

**IO_uring Implementation:**
```bash
cd poc-iouring-minimal
make
```

### Running the Benchmark

After building all components, you can run the benchmark script to compare the different implementations:

```bash
python benchmark.py
```

The benchmark will test Redis performance with different durability settings and BPF extensions.

## Implementation Details

Each approach is located in its own directory:

- `batch_process/`: Batch processing implementation
- `poc-iouring-minimal/`: io_uring batched I/O implementation
- `delayed-fsync/`: Delayed fsync implementation
- `fsync-fast-notify/`: Fast-path optimization implementation
- `bpf-sync-kernel/`: Kernel-level sync optimization

Each implementation directory contains:
- C source files for the BPF program
- A Makefile for building
- A README with implementation-specific details

## Troubleshooting

If you encounter build issues:

1. Make sure all prerequisites are installed
2. Check that libbpf and kernel headers are available
3. Review the error messages in the build output
4. Consult the implementation-specific README files for any special requirements

## Directory Structure

```
redis-durability-tuning/
├── batch_process/         # Batch process implementation
├── delayed-fsync/         # Delayed fsync implementation
├── fsync-fast-notify/     # Fast path optimization implementation
├── poc-iouring-minimal/   # IO_uring implementation
├── redis/                 # Modified Redis source code
├── benchmark.py           # Benchmark script
└── Makefile               # Main Makefile for building components
```

