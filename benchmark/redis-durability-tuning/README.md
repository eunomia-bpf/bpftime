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

## Building and Running

1. Clone the repository and build Redis:
   ```
   make redis
   ```

2. Run the benchmark script to compare different approaches:
   ```
   python benchmark.py
   ```

## Implementation Details

Each approach is located in its own directory:

- `poc-iouring-minimal/`: io_uring batched I/O implementation
- `delayed-fsync/`: Delayed fsync implementation
- `fsync-fast-notify/`: Fast-path optimization implementation
- `bpf-sync-kernel/`: Kernel-level sync optimization

Each implementation directory contains its own README with more detailed information.

