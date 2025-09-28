# BPF Bloom Filter Test Guide

This document describes the implementation and testing of BPF Bloom Filter (`BPF_MAP_TYPE_BLOOM_FILTER`) functionality in `bpftime`. Bloom filters are probabilistic data structures used for set membership testing with no false negatives but possible false positives.

## Components

- `uprobe_bloom_filter.bpf.c`: eBPF kernel program defining and operating `BPF_MAP_TYPE_BLOOM_FILTER`
- `uprobe_bloom_filter.c`: User-space program loading eBPF and monitoring user access patterns (includes embedded BPF program via skeleton)
- `target.c`: Demo program whose functions are monitored by eBPF uprobe
- `test_correct_bloom_filter.sh`: Automated test script for demonstrating bloom filter behavior

## Quick Start

### 1. Build Components
```bash
make clean
make uprobe_bloom_filter target
```

### 2. Run Bloom Filter Demo
```bash
# Recommended: Use script (automatically sets VM type)
./test_correct_bloom_filter.sh

# Manual execution (requires two terminals):
# Terminal 1: Start bloom filter monitor (bpftime Server)
BPFTIME_VM_NAME=ubpf LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_bloom_filter

# Terminal 2: Start target program (bpftime Agent/Client)
BPFTIME_VM_NAME=ubpf LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target
```

**Note**: This demo requires `BPFTIME_VM_NAME=ubpf` environment variable to use the ubpf VM instead of the default llvm VM for compatibility.

## Bloom Filter Behavior Verification

### Test Principle
The demo monitors user access patterns using a bloom filter:
1. When `user_access()` function is called with a user ID
2. eBPF program checks if the user ID exists in the bloom filter
3. If not found (miss): Mark as new user and add to bloom filter
4. If found (hit): Mark as repeat user (possible false positive)

### Expected Statistics
```text
=== Bloom Filter Real-time Monitoring Statistics ===
Total user accesses:      10
New users (first access): 8
Repeat users (re-access): 2
Admin operations:         4
System events:            6

--- Bloom Filter Performance Analysis ---
Bloom Filter hits:        2 (detected as possibly existing)
Bloom Filter misses:      8 (definitely not existing)
Hit rate:                 20.00%
New user ratio:           80.00%
Repeat user ratio:        20.00%
```

### Sample BPF Program Output
```text
Bloom filter MISS for user_id=1003 (new user, added to filter)
Bloom filter HIT for user_id=1003 (repeat user)
Admin operation by admin_id=2002
System event triggered
```

### Verification Points
- **No False Negatives**: All new users are correctly identified ✅
- **Possible False Positives**: Some new users may be misjudged as repeat users
- **Statistical Consistency**: new users + repeat users = total accesses
- **False Positive Rate**: Estimated based on bloom filter hits vs actual repeat users

## Technical Implementation

### BPF Map Definition
The embedded BPF program defines the bloom filter map as:
```c
struct {
    __uint(type, BPF_MAP_TYPE_BLOOM_FILTER);
    __uint(max_entries, 1000);
    __uint(value_size, sizeof(u32));
    __uint(map_extra, 3); // Number of hash functions
} user_bloom_filter SEC(".maps");
```

### Core BPF Operations
- **Lookup**: `bpf_map_lookup_elem(&user_bloom_filter, &user_id)`
- **Update**: `bpf_map_update_elem(&user_bloom_filter, &user_id, &value, BPF_ANY)`
- **Peek**: `bpf_map_peek_elem(&user_bloom_filter, &user_id)` (userspace)

### Statistics Tracking
```c
// Statistics index definitions
#define STAT_TOTAL_ACCESSES 0
#define STAT_UNIQUE_USERS 1
#define STAT_REPEAT_USERS 2
#define STAT_BLOOM_HITS 5
#define STAT_BLOOM_MISSES 6
```

### User Access Pattern
The target program simulates various user access patterns:
- **High frequency**: User access with rotating user IDs (1001-1010)
- **Medium frequency**: Admin operations
- **Low frequency**: System events

## Key Bloom Filter Characteristics

### 1. No False Negatives
- If bloom filter returns "not found", the element is **definitely not** in the set
- All new users are correctly identified as new

### 2. Possible False Positives
- If bloom filter returns "found", the element **might be** in the set
- Some new users may be incorrectly identified as repeat users
- False positive rate depends on:
  - Number of elements added
  - Bloom filter size
  - Number of hash functions

### 3. Memory Efficiency
- Uses fixed-size bit array regardless of number of elements
- Much more memory-efficient than hash tables for large datasets
- Trade-off: accuracy vs memory usage

## Performance Analysis

The demo provides real-time analysis of bloom filter performance:

### Metrics Tracked
- **Total Accesses**: Number of `user_access()` calls
- **New Users**: Users seen for the first time (bloom filter miss → add)
- **Repeat Users**: Users seen before (bloom filter hit)
- **Hit Rate**: Percentage of bloom filter hits
- **False Positive Estimate**: Estimated false positive occurrences

### Analysis Output
```text
=== Bloom Filter Test Analysis ===
Test result verification:
  Theory: new users + repeat users = total accesses
  Actual: 8 + 2 = 10 (total accesses: 10)
  ✅ Consistency check passed

  Bloom Filter characteristics verification:
  - No false negatives: all new users correctly identified ✅
  - Possible false positives: some new users may be misjudged as repeat users
  - False positive detection: possible 0 false positives
```

## Troubleshooting

### VM Compatibility Issues
If you see errors like "No VM factory registered for name: llvm", make sure to set:
```bash
export BPFTIME_VM_NAME=ubpf
```

### No Events Captured
If statistics remain at 0, check:
1. VM type is set correctly (`BPFTIME_VM_NAME=ubpf`)
2. Target program is running with the correct LD_PRELOAD
3. uprobe attachment succeeded (check console output)

## Verified bpftime Features

- `BPF_MAP_TYPE_BLOOM_FILTER`: Bloom filter map creation and management
- `bpf_map_lookup_elem()`: Membership testing in bloom filters
- `bpf_map_update_elem()`: Adding elements to bloom filters
- `bpf_map_peek_elem()`: Userspace bloom filter querying
- Uprobe integration with bloom filter maps
- Real-time bloom filter performance monitoring
- False positive rate analysis and estimation
- ubpf VM integration and compatibility 