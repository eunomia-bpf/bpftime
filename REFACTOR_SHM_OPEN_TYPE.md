# Refactoring Plan: Make `shm_open_type` Internal

## Executive Summary

This document outlines a plan to make `shm_open_type` an internal implementation detail by introducing a clean public API for querying process role. The goal is to eliminate direct `shm_open_type` usage outside of initialization and core shared memory code, making the codebase more robust and maintainable.

## Current State Analysis

### 1. **Enum Definition** (`runtime/include/bpftime_shm.hpp:125-130`)

```cpp
enum class shm_open_type {
    SHM_REMOVE_AND_CREATE,  // Server mode: creates new shared memory
    SHM_OPEN_ONLY,          // Agent mode: opens existing shared memory
    SHM_NO_CREATE,          // Test mode: no shared memory created
    SHM_CREATE_OR_OPEN,     // Hybrid mode: creates or opens
};
```

### 2. **Current Usage Patterns**

Based on code analysis, `shm_open_type` is used in three main patterns:

#### Pattern A: Direct Comparison (Scattered throughout codebase)
```cpp
if (shm_holder.global_shared_memory.get_open_type() == shm_open_type::SHM_OPEN_ONLY)
if (open_type != shm_open_type::SHM_REMOVE_AND_CREATE)
```

**Locations:**
- `runtime/src/bpftime_shm_internal.cpp:73-74` - Agent cleanup in destructor
- `runtime/src/bpftime_shm_internal.cpp:909` - CUDA host memory registration
- `runtime/src/bpftime_shm_internal.cpp:934` - Destructor early return
- `runtime/src/bpf_map/gpu/nv_gpu_array_map.cpp:17-18` - GPU array map initialization
- `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp:27-28` - GPU ringbuf map initialization
- `runtime/src/bpf_map/gpu/nv_gpu_shared_array_map.cpp:23-24,71-72` - GPU shared array map operations

#### Pattern B: Initialization with Specific Type (Entry points)
```cpp
bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
```

**Locations:**
- **Server/Creator Mode:**
  - `runtime/syscall-server/syscall_server_utils.cpp:49` - Syscall server
  - `daemon/user/bpftime_driver.cpp:383` - Daemon driver
  - `runtime/unit-test/tailcall/test_user_to_kernel_tailcall.cpp:71` - Tests
  - `runtime/unit-test/test_bpftime_shm_json.cpp:61` - Tests

- **Agent/Client Mode:**
  - `runtime/agent/agent.cpp:165` - Main agent entry point
  - `tools/cli/main.cpp:325` - CLI detach command
  - `tools/aot/main.cpp:108,159` - AOT compilation tool
  - `tools/bpftimetool/main.cpp:83,177,266` - Various bpftimetool commands
  - `runtime/test/src/test_shm_client.cpp:30` - Test client
  - `runtime/unit-test/test_bpftime_shm_json.cpp:95` - Tests

- **Create-or-Open Mode:**
  - `tools/bpftimetool/main.cpp:164,189` - Import commands

#### Pattern C: Switch-Case Logic (Initialization)
```cpp
if (type == shm_open_type::SHM_OPEN_ONLY) { ... }
else if (type == shm_open_type::SHM_CREATE_OR_OPEN) { ... }
else if (type == shm_open_type::SHM_REMOVE_AND_CREATE) { ... }
else if (type == shm_open_type::SHM_NO_CREATE) { ... }
```

**Location:**
- `runtime/src/bpftime_shm_internal.cpp:631-730` - Shared memory initialization

### 3. **Identified Issues**

#### Issue #1: **`shm_open_type` is Public API**
**Problem:** `shm_open_type` enum is defined in the public header `bpftime_shm.hpp` and used directly throughout the codebase for runtime checks. This exposes internal implementation details.

**Impact:**
- External users depend on internal implementation
- Hard to refactor initialization logic
- Breaking changes affect external code

#### Issue #2: **Magic Value Comparisons**
**Problem:** Direct comparisons to enum values are scattered throughout the codebase, making it hard to understand intent and easy to introduce bugs.

**Example:**
```cpp
// What does this really mean? Is it checking for agent mode? Server mode? Neither?
if (shm_holder.global_shared_memory.get_open_type() != shm_open_type::SHM_REMOVE_AND_CREATE)
```

**Impact:** Low readability, high maintenance burden, prone to logical errors.

#### Issue #3: **Inconsistent Intent Checking**
**Problem:** Same logical intent expressed differently in different places.

**Examples:**
```cpp
// Agent check - three different patterns:
open_type == shm_open_type::SHM_OPEN_ONLY                        // Pattern 1
open_type != shm_open_type::SHM_REMOVE_AND_CREATE                // Pattern 2 (incorrect!)
shm_holder.global_shared_memory.get_open_type() == SHM_OPEN_ONLY // Pattern 3
```

**Impact:** Pattern 2 is **incorrect** - it would match `SHM_CREATE_OR_OPEN` and `SHM_NO_CREATE`, not just agent mode.

#### Issue #4: **Mixed Concerns**
**Problem:** `shm_open_type` serves dual purpose:
1. **Initialization mode** (how to open shared memory)
2. **Process role** (agent vs server vs test)

**Impact:** Conflation of concerns makes code harder to reason about.

## Proposed Solution

### Step 1: Add Public Role-Based API

#### 1.1 Add High-Level Role Query Functions
**Location:** `runtime/include/bpftime_shm.hpp`

```cpp
// Already implemented:
int bpftime_is_agent();   // Returns 1 if SHM_OPEN_ONLY
int bpftime_is_server();  // Returns 1 if SHM_REMOVE_AND_CREATE

// Additional functions to add:
int bpftime_is_test_mode();           // Returns 1 if SHM_NO_CREATE
int bpftime_is_hybrid_mode();         // Returns 1 if SHM_CREATE_OR_OPEN
int bpftime_shm_initialized();        // Returns 1 if shm is initialized
```

**Implementation:** `runtime/src/bpftime_shm.cpp`

```cpp
int bpftime_is_test_mode()
{
    return shm_holder.global_shared_memory.get_open_type() ==
                   shm_open_type::SHM_NO_CREATE ? 1 : 0;
}

int bpftime_is_hybrid_mode()
{
    return shm_holder.global_shared_memory.get_open_type() ==
                   shm_open_type::SHM_CREATE_OR_OPEN ? 1 : 0;
}

int bpftime_shm_initialized()
{
    return shm_holder.global_shared_memory.get_open_type() !=
                   shm_open_type::SHM_NO_CREATE &&
           shm_holder.global_shared_memory.get_manager() != nullptr ? 1 : 0;
}
```

**Benefits:**
- Clear intent in calling code
- Single source of truth for role determination
- Encapsulates internal `shm_open_type` logic

### Step 2: Move `shm_open_type` to Internal Header

#### 2.1 Move Enum Definition
**From:** `runtime/include/bpftime_shm.hpp` (public header)
**To:** `runtime/src/bpftime_shm_internal.hpp` (internal header)

This makes `shm_open_type` an implementation detail that external users cannot access.

#### 2.2 Keep Initialization Function Public
Keep `bpftime_initialize_global_shm()` in the public API, but update signature:

**Option A: Use role-based initialization**
```cpp
// New public API
void bpftime_initialize_global_shm_as_server();
void bpftime_initialize_global_shm_as_agent();
void bpftime_initialize_global_shm_hybrid();
void bpftime_initialize_global_shm_test_mode();
```

**Option B: Keep existing but mark internal**
```cpp
// Keep current signature but document as internal
// Internal use only - use role-based init functions instead
void bpftime_initialize_global_shm(bpftime::shm_open_type type);
```

**Recommendation:** Start with Option B to maintain backward compatibility, then gradually migrate to Option A.

### Step 3: Replace All Direct `shm_open_type` Comparisons

#### 3.1 Update Internal Runtime Code
Replace scattered comparisons with the new API:

**Files to update:**
- `runtime/src/bpftime_shm_internal.cpp` (3 locations)
- `runtime/src/bpf_map/gpu/*.cpp` (6 locations)

**Example transformation:**
```cpp
// Before:
if (shm_holder.global_shared_memory.get_open_type() == shm_open_type::SHM_OPEN_ONLY) {
    // Agent-specific code
}

// After:
if (bpftime_is_agent()) {
    // Agent-specific code
}
```

#### 3.2 Update Tool Initialization Code
Update initialization calls in tools to use role-based semantics:

**Files to update:**
- `tools/cli/main.cpp:325`
- `tools/aot/main.cpp:108,159`
- `tools/bpftimetool/main.cpp:83,177,266`
- And other tool entry points

**Example:**
```cpp
// Before:
bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);

// After (keeping backward compatibility):
bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
// Or using new API (future):
bpftime_initialize_global_shm_as_agent();
```

### Step 4: Update Public API Header

Remove `shm_open_type` and `global_shm_open_type` from `bpftime_shm.hpp`:

```cpp
// REMOVE these from public header:
enum class shm_open_type { ... };
extern const shm_open_type global_shm_open_type;

// KEEP these in public header:
int bpftime_is_agent();
int bpftime_is_server();
int bpftime_is_test_mode();
int bpftime_is_hybrid_mode();
int bpftime_shm_initialized();

// Keep initialization function (mark as internal in docs)
void bpftime_initialize_global_shm(bpftime::shm_open_type type);
```

## Implementation Plan

### Sprint 1: Quick Wins (1-2 days)
- [x] Add `bpftime_is_agent()` and `bpftime_is_server()` ✓ Already done
- [ ] Add `bpftime_is_test_mode()` and `bpftime_is_hybrid_mode()`
- [ ] Replace all direct comparisons in `bpftime_shm_internal.cpp` with new API
- [ ] Add unit tests for new API functions

### Sprint 2: GPU Map Consolidation (2-3 days)
- [ ] Create `nv_gpu_map_base` class with common logic
- [ ] Refactor `nv_gpu_array_map_impl` to use base class
- [ ] Refactor `nv_gpu_ringbuf_map_impl` to use base class
- [ ] Refactor `nv_gpu_shared_array_map_impl` to use base class
- [ ] Add tests for GPU map functionality

### Sprint 3: Role Separation (2-3 days)
- [ ] Introduce `bpftime_process_role` enum
- [ ] Implement `bpftime_get_process_role()` API
- [ ] Update documentation to recommend role API over `shm_open_type`
- [ ] Add migration guide for external users

### Sprint 4: Defensive Checks (1-2 days)
- [ ] Add `bpftime_shm_initialized()` check
- [ ] Add initialization checks to public APIs
- [ ] Add role validation to operation-specific functions
- [ ] Update error messages for clarity

## Testing Strategy

### Unit Tests
```cpp
TEST_CASE("Role detection APIs") {
    SECTION("Server mode") {
        bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
        REQUIRE(bpftime_is_server() == 1);
        REQUIRE(bpftime_is_agent() == 0);
        REQUIRE(bpftime_is_test_mode() == 0);
    }

    SECTION("Agent mode") {
        bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
        REQUIRE(bpftime_is_agent() == 1);
        REQUIRE(bpftime_is_server() == 0);
    }

    SECTION("Before initialization") {
        REQUIRE(bpftime_shm_initialized() == 0);
        REQUIRE(bpftime_get_process_role() == ROLE_UNKNOWN);
    }
}
```

### Integration Tests
- Test agent connecting to server
- Test CUDA GPU maps in agent mode
- Test hybrid mode (create-or-open)
- Test error paths with defensive checks

## Migration Guide for Downstream Users

### For Internal Code
**Before:**
```cpp
if (shm_holder.global_shared_memory.get_open_type() == shm_open_type::SHM_OPEN_ONLY) {
    // agent logic
}
```

**After:**
```cpp
if (bpftime_is_agent()) {
    // agent logic
}
```

### For External Code
If external code uses `shm_open_type` directly, provide deprecation warnings and migration timeline:

1. **Phase 1 (Current):** Both APIs available
2. **Phase 2 (Next release):** Deprecation warnings for direct `shm_open_type` usage
3. **Phase 3 (Future release):** `shm_open_type` becomes internal-only

## Risk Assessment

### Low Risk Changes
- Adding new API functions (backward compatible)
- Replacing internal comparisons with API calls
- Adding unit tests

### Medium Risk Changes
- GPU map refactoring (requires careful CUDA testing)
- Adding defensive checks (might expose existing bugs)

### High Risk Changes
- Changing `shm_open_type` enum or semantics
- Breaking changes to public API

## Success Metrics

- **Code Quality:**
  - Reduce direct `shm_open_type` comparisons by 90%+
  - Reduce GPU map code duplication by 70%+
  - Increase code coverage for shared memory initialization by 20%+

- **Developer Experience:**
  - Clearer error messages when SHM not initialized
  - Faster onboarding (easier to understand role vs initialization mode)
  - Fewer bugs related to incorrect role detection

- **Maintainability:**
  - Single source of truth for role detection logic
  - GPU map logic consolidated in one place
  - Clear separation between initialization and runtime concerns

## Conclusion

This refactoring plan addresses the key issues with `shm_open_type` usage:
1. **Magic value comparisons** → Semantic API functions
2. **Inconsistent intent** → Single source of truth
3. **GPU code duplication** → Base class consolidation
4. **Lack of abstraction** → Role-based public API
5. **Mixed concerns** → Separate initialization from role
6. **Scattered CUDA logic** → Centralized GPU map base class

The phased approach allows for incremental improvement with low risk, while the comprehensive scope ensures long-term maintainability and robustness.
