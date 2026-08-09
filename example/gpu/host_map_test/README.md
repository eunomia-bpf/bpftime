# Host Map Test - GPU Host-backed Map Types

## Overview

This example demonstrates and tests two GPU map types that store data in **Host memory** (as opposed to GPU device memory):

| Map Type | ID | Description |
|----------|-----|-------------|
| `BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP` | 1512 | Per-GPU-thread storage in Host memory |
| `BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP` | 1513 | Shared storage in Host memory |

## Why Host-backed Maps?

### Comparison with GPU-memory Maps

| Feature | GPU Memory Maps | Host Memory Maps |
|---------|-----------------|------------------|
| Storage Location | GPU Device Memory | Host (CPU) Memory |
| GPU Access Speed | Fast | Slower (PCIe) |
| Host Access Speed | Slower (DMA) | Fast |
| Memory Size | Limited by GPU VRAM | Limited by Host RAM |
| Use Case | High-frequency GPU updates | Frequent Host reads |

### When to Use Host-backed Maps

1. **Frequent Host Reads**: When userspace needs to frequently read map data
2. **Large Data Storage**: When data exceeds GPU memory limits
3. **Debugging/Tracing**: When low-latency Host access is more important than GPU write speed
4. **Cross-device Sharing**: When data needs to be shared across multiple GPUs

## Map Types Explained

### BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP (1513)

**Shared storage** - All GPU threads see and modify the same values.

```
Memory Layout:
┌────────────────────────────────────────┐
│ Key 0 │ Key 1 │ Key 2 │ Key 3 │ ...   │  <- Single copy
└────────────────────────────────────────┘
   ↑        ↑        ↑
   │        │        │
   └────────┴────────┴── All threads read/write here
```

**Characteristics**:
- Simple lookup returns single value per key
- Last writer wins (non-atomic writes)
- Good for global counters, configuration values
- Contention under heavy concurrent writes

### BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP (1512)

**Per-thread storage** - Each GPU thread has its own independent slot.

```
Memory Layout:
┌──────────────────────────────────────────────────────┐
│ Thread 0 │ Thread 1 │ Thread 2 │ ... │ Thread N-1   │ <- Key 0
├──────────────────────────────────────────────────────┤
│ Thread 0 │ Thread 1 │ Thread 2 │ ... │ Thread N-1   │ <- Key 1
├──────────────────────────────────────────────────────┤
│    ...   │    ...   │    ...   │ ... │    ...       │
└──────────────────────────────────────────────────────┘
```

**Characteristics**:
- Lookup returns `value_size * thread_count` bytes (array)
- No contention between threads
- Good for per-thread statistics, timestamps, counters
- Higher memory usage (multiplied by thread count)

## Building

```bash
# From bpftime root directory
make -C example/gpu/host_map_test
```

You can customize the number of map entries by defining `HOST_MAP_MAX_ENTRIES`:

```bash
# Use 20 entries instead of the default 10
make -C example/gpu/host_map_test HOST_MAP_MAX_ENTRIES=20
```

Requirements:
- bpftime built with CUDA support (`-DBPFTIME_ENABLE_CUDA_ATTACH=1`)
- CUDA toolkit installed

## Running

### Terminal 1: Start the BPF program (Server)

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/host_map_test/host_map_test [thread_count]
```

Optional argument `thread_count` specifies how many GPU threads to monitor (default: 16).

### Terminal 2: Run the CUDA application (Agent)

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/host_map_test/vec_add [threads_per_block] [num_blocks] [sleep_ms]
```

Arguments:
- `threads_per_block`: Threads per CUDA block (default: 10)
- `num_blocks`: Number of CUDA blocks (default: 1)
- `sleep_ms`: Sleep between iterations in ms (default: 1000)

### Example

```bash
# Terminal 1: Monitor 20 threads
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/host_map_test/host_map_test 20

# Terminal 2: Run with 10 threads/block, 2 blocks = 20 total threads
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/host_map_test/vec_add 10 2 500
```

## Example Output

```
========== 12:34:56 ==========

[shared_counter - GPU_ARRAY_HOST_MAP]
  All GPU threads share these counters (last-writer-wins)
  Key        Value
  ---        -----
  0          15
  1          12
  2          18
  3          14

[perthread_counter - PERGPUTD_ARRAY_HOST_MAP]
  Each GPU thread has independent storage

  Key 0 (call_count):
    Thread     Value
    ------     -----
    0          5
    1          5
    2          5
    3          5
    ---
    Total: 20 (from 4 active threads)

  Key 1 (exec_time_ns):
    Thread     Value
    ------     -----
    0          1234
    1          1256
    2          1189
    3          1312
    ---
    Total: 4991 (from 4 active threads)

  Key 2 (thread_id):
    Thread     Value
    ------     -----
    0          0
    1          1
    2          2
    3          3
    ---
    Total: 6 (from 4 active threads)
```

## Code Structure

- **`host_map_test.bpf.c`**: eBPF program with map definitions and probes
  - Uses `HOST_MAP_MAX_ENTRIES` macro (default: 10) for map max_entries
  - `perthread_counter`: PERGPUTD_ARRAY_HOST_MAP for per-thread statistics
  - `shared_counter`: GPU_ARRAY_HOST_MAP for shared counters
  - `thread_timestamp`: PERGPUTD_ARRAY_HOST_MAP for entry timestamps

- **`host_map_test.c`**: Userspace program that reads and displays map data
  - Also defines `HOST_MAP_MAX_ENTRIES` macro for consistency
  - Uses `bpf_map_get_next_key()` to dynamically iterate through map keys

- **`vec_add.cu`**: CUDA vector addition program to trigger probes

## Understanding the Output

### shared_counter (GPU_ARRAY_HOST_MAP)

Shows global counters incremented by all threads. Due to non-atomic writes and last-writer-wins semantics, the sum may be less than total thread executions.

### perthread_counter (PERGPUTD_ARRAY_HOST_MAP)

Shows per-thread data:
- **Key 0 (call_count)**: How many times each thread executed the kernel
- **Key 1 (exec_time_ns)**: Last execution time for each thread
- **Key 2 (thread_id)**: Thread ID stored by each thread (for verification)

## Use Cases

### 1. Performance Profiling

Track per-thread execution times to identify slow threads or load imbalance.

### 2. Call Counting

Count kernel invocations per thread without contention issues.

### 3. Debugging

Store thread-specific debug data that can be read from userspace with low latency.

### 4. Configuration Distribution

Use shared map to distribute configuration to all GPU threads.

## Troubleshooting

**No data appears**:
- Check that both processes are running
- Verify CUDA kernel name matches the SEC annotation
- Check bpftime logs for attachment errors

**Partial data**:
- Ensure thread_count argument matches actual GPU thread count
- Some threads may not have executed yet

**Incorrect counts in shared_counter**:
- Expected with non-atomic writes
- Use per-thread map for accurate counting, aggregate in userspace

## Related Examples

- `gpu_shard_array`: Tests GPU_ARRAY_MAP (GPU memory backed)
- `threadhist`: Uses PERGPUTD_ARRAY_MAP for thread histograms
- `cuda-counter`: Basic probe/retprobe example
