# BPF Queue (FIFO) Test Guide

This document describes the implementation and testing of BPF Queue (`BPF_MAP_TYPE_QUEUE`) functionality in `bpftime`. BPF Queue follows First In First Out (FIFO) data processing.

## Core Feature: FIFO (First In First Out)

- **Processing Order**: Events are processed in the same order as their entry
- **Pop Behavior**: The first (oldest) event pushed to queue is popped first
- **Use Cases**: Suitable for event streaming, task scheduling, and sequential data processing

## Components

- `uprobe_queue.bpf.c`: eBPF kernel program defining and operating `BPF_MAP_TYPE_QUEUE`
- `uprobe_queue.c`: User-space program loading eBPF and reading events from queue
- `target.c`: Demo program whose functions are monitored by eBPF uprobe
- `run_demo.sh`: Automated test script for demonstrating FIFO behavior

## Quick Start

### 1. Build Components
```bash
make clean
make uprobe_queue target
```

### 2. Run Queue Demo
```bash
# Recommended: Use script
./run_demo.sh -r

# Manual execution (requires two terminals):
# Terminal 1: Start queue monitor (bpftime Server)
LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue

# Terminal 2: Start target program (bpftime Agent/Client)
LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target
```

## FIFO Behavior Verification

### Example Scenario
If `target.c` triggers events in this order (identified by counter field):
1. Event A (counter 1) pushed to queue
2. Event B (counter 2) pushed to queue
3. Event C (counter 3) pushed to queue

When popping from queue, the order will be:
1. Event A (counter 1) - first pushed, first popped
2. Event B (counter 2)
3. Event C (counter 3) - last pushed, last popped

### Expected Log Output
```text
// BPF program output (bpf_printk)
Pushed event to queue: pid=xxx, counter=1, ...
Pushed event to queue: pid=xxx, counter=2, ...
Pushed event to queue: pid=xxx, counter=3, ...

// User-space program output (uprobe_queue)
Queue status: non-empty (head event: function_id=X, counter=1)
[HH:MM:SS.ms] target_function() called - counter:1 ...
[HH:MM:SS.ms] target_function() called - counter:2 ...
[HH:MM:SS.ms] target_function() called - counter:3 ...
Processed 3 events this round
```

## Technical Implementation

### BPF Map Definition
```c
struct {
    __uint(type, BPF_MAP_TYPE_QUEUE);
    __uint(max_entries, 64);
    __type(value, struct event_data);
} events_queue SEC(".maps");
```

### Core BPF Operations
- **Push**: `bpf_map_push_elem(&events_queue, &event, BPF_ANY)`
- **Pop**: User-space uses `BPF_MAP_LOOKUP_AND_DELETE_ELEM` syscall
- **Peek**: User-space uses `BPF_MAP_LOOKUP_ELEM` syscall

### Event Data Structure
```c
struct event_data {
    uint64_t timestamp;
    uint32_t pid;
    uint32_t tid;
    uint32_t counter;
    uint32_t function_id;
    int32_t input_value;
    char comm[16];
};
```

## Verified bpftime Features

- `BPF_MAP_TYPE_QUEUE`: Queue map creation and management
- `bpf_map_push_elem()`: Push operation on queue maps
- `BPF_MAP_LOOKUP_AND_DELETE_ELEM`: Pop operation on queue maps
- `BPF_MAP_LOOKUP_ELEM`: Peek operation on queue maps
- FIFO data access pattern
- Uprobe integration with queue maps

## Common Use Cases

- Event streaming and processing
- Task scheduling and job queues
- Sequential data processing
- Message passing between components
- Rate limiting and buffering

## Queue vs Stack Comparison

| Feature          | Queue (FIFO)          | Stack (LIFO)          |
| ---------------- | --------------------- | --------------------- |
| Processing Order | First In First Out    | Last In First Out     |
| Use Case         | Sequential processing | Recent data priority  |
| Pop Behavior     | Oldest event first    | Newest event first    |
| Example          | Event streaming       | Function call tracing |

## Troubleshooting

1. **Environment Check**:
   - Ensure commands run in `example/queue_demo/` directory
   - Verify `bpftime` libraries exist in `../../build/runtime/`

2. **Build Verification**:
   - Run `make build` in project root
   - Check `libbpftime-syscall-server.so` and `libbpftime-agent.so` exist

3. **Target Program**:
   - Ensure `target` compiled with `make uprobe_queue target`
   - Test `./target` runs independently

4. **FIFO Verification**:
   - Compare push order in `bpf_printk` with pop order in user-space output
   - Counter field should appear in same order during pop operations

5. **Edge Cases**:
   - Empty queue operations should return `-ENOENT`
   - Full queue operations should return appropriate error codes 