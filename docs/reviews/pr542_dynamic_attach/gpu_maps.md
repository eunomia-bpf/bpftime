# GPU Map Review Notes

Files:

- `runtime/src/bpf_map/gpu/nv_gpu_shared_array_map.cpp`
- `runtime/src/bpf_map/gpu/nv_gpu_shared_array_map.hpp`
- `runtime/src/bpf_map/gpu/nv_gpu_per_thread_array_map.cpp` (minor changes)

## Observations

- The updated `nv_gpu_shared_array_map_impl::try_initialize_for_agent_and_get_mapped_address()` now:
  - Calls `cuInit(0)` and probes device contexts to open the CUDA IPC handle.
  - Creates a private non-blocking stream for agent-side async memcpy to avoid default-stream deadlocks.
  - Adds guard checks in `elem_lookup/elem_update` to return `EINVAL` if mapping/context is unavailable.

These are directionally reasonable for “late attach / unknown current context” scenarios.

## Should fix

### M1. `map_get_next_key` assigns to `next_key` pointer instead of output value

- **Location**: `runtime/src/bpf_map/gpu/nv_gpu_shared_array_map.cpp:190-210`
- **Problem**:
  - When `key_val >= max_entries`, code does `next_key = 0;` which only modifies the local pointer variable, not the caller-visible output.
- **Impact**: Caller may observe uninitialized/previous `next_key` value; iteration semantics become inconsistent.
- **Recommended fix**:
  - Use `next_key_val = 0;` (or return `-ENOENT` / set `errno` per project convention).

## Optional

### M2. Agent-side device/context probing behavior should be documented

- **Location**: `nv_gpu_shared_array_map_impl::try_initialize_for_agent_and_get_mapped_address()`
- **Suggestion**:
  - Document the selection order (env `BPFTIME_CUDA_DEVICE` -> all devices -> device 0).
  - Document that primary contexts may be retained temporarily and how/when they are released (if applicable).

