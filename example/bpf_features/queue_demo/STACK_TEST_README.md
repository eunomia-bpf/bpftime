# BPF Stack (LIFO) Test Guide

This document describes the implementation and testing of BPF Stack (`BPF_MAP_TYPE_STACK`) functionality in `bpftime`. BPF Stack follows Last In First Out (LIFO) data processing.

## Components

- `uprobe_stack.bpf.c`: eBPF kernel program defining and operating `BPF_MAP_TYPE_STACK`
- `uprobe_stack.c`: User-space program loading eBPF and reading events from stack
- `target.c`: Demo program whose functions are monitored by eBPF uprobe
- `run_stack_demo.sh`: Automated test script for demonstrating LIFO behavior

## Quick Start

### 1. Build Components
```bash
make clean
make uprobe_stack target
```

### 2. Run Stack Demo
```bash
# Recommended: Use script
./run_stack_demo.sh -s

# Manual execution (requires two terminals):
# Terminal 1: Start stack monitor (bpftime Server)
LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack

# Terminal 2: Start target program (bpftime Agent/Client)
LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target
```

## LIFO Behavior Verification

### Example Scenario
If `target.c` triggers events in this order (identified by counter field):
1. Event A (counter 1) pushed to stack
2. Event B (counter 2) pushed to stack  
3. Event C (counter 3) pushed to stack

When popping from stack, the order will be:
1. Event C (counter 3) - last pushed, first popped
2. Event B (counter 2)
3. Event A (counter 1) - first pushed, last popped

### Expected Log Output
```text
// BPF program output (bpf_printk)
Pushed event to stack: pid=xxx, counter=1, ...
Pushed event to stack: pid=xxx, counter=2, ...
Pushed event to stack: pid=xxx, counter=3, ...

// User-space program output (uprobe_stack)
Stack status: non-empty (top event: function_id=Y, counter=3)
=== Starting to pop events from stack (LIFO order) ===
[HH:MM:SS.ms] [stack pop #1] ... counter:3 ...
[HH:MM:SS.ms] [stack pop #2] ... counter:2 ...
[HH:MM:SS.ms] [stack pop #3] ... counter:1 ...
=== Stack pop completed, processed 3 events ===
```

## Technical Implementation

### BPF Map Definition
```c
struct {
    __uint(type, BPF_MAP_TYPE_STACK);
    __uint(max_entries, 64);
    __type(value, struct event_data);
} events_stack SEC(".maps");
```

### Core BPF Operations
- **Push**: `bpf_map_push_elem(&events_stack, &event, BPF_ANY)`
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

- `BPF_MAP_TYPE_STACK`: Stack map creation and management
- `bpf_map_push_elem()`: Push operation on stack maps
- `BPF_MAP_LOOKUP_AND_DELETE_ELEM`: Pop operation on stack maps
- `BPF_MAP_LOOKUP_ELEM`: Peek operation on stack maps
- LIFO data access pattern
- Uprobe integration with stack maps
