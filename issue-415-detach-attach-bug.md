# bpftime Detach/Re-attach Issue #415

## Environment
- **OS**: Debian 12
- **Compiler**: clang 14
- **LLVM**: version 14
- **bpftime**: Latest version from master branch (unmodified)

## Problem Description

After performing a `bpftime detach` operation, subsequent `bpftime attach` operations to the same process fail silently. The issue appears to be that the agent state in the target process is not properly cleaned up during detach, causing future attach attempts to skip initialization with the message "Agent already initialized, skipping re-initializing.." but without actually functioning.

**Note**: This issue was reproduced using the unmodified malloc example from `bpftime/example/malloc`.

## Steps to Reproduce

1. **Terminal 1**: Start bpftime load
   ```bash
   bpftime load ./example/malloc/malloc
   ```
   Output shows normal timestamp logging:
   ```
   18:12:03 
   18:12:04 
   18:12:05 
   18:12:06 
   ```

2. **Terminal 2**: Start target process
   ```bash
   ./example/malloc/victim & echo $!
   ```
   Process ID returned: `608542`

3. **Terminal 3**: First attach (works correctly)
   ```bash
   bpftime attach 608542
   ```
   
   **Logs from first attach:**
   ```
   [2025-07-09 18:12:10][info][609037] bpf_attach_ctx constructed
   [2025-07-09 18:12:10][info][609037] Register attach-impl defined helper bpf_get_func_arg, index 183
   [2025-07-09 18:12:10][info][609037] Register attach-impl defined helper bpf_get_func_ret_id, index 184
   [2025-07-09 18:12:10][info][609037] Register attach-impl defined helper bpf_get_retval, index 186
   [2025-07-09 18:12:10][info][609037] Initializing agent..
   [2025-07-09 18:12:11][info][609037] Executable path: /home/qianshuo01/Project/bpftime/example/malloc/victim
   [2025-07-09 18:12:11][info][609037] Main initializing for handlers done, try to initialize cuda link handles....
   [2025-07-09 18:12:11][info][609037] Attach successfully
   ```
   
   **Terminal 1 now shows successful monitoring:**
   ```
   18:12:13 
           pid=608542      malloc calls: 10
   18:12:14 
           pid=608542      malloc calls: 10
   18:12:15 
           pid=608542      malloc calls: 10
   ```

4. **Terminal 3**: Detach
   ```bash
   bpftime detach
   ```
   
   **Detach log:**
   ```
   [2025-07-09 18:12:20][info][608542] Detaching..
   ```
   
   **Terminal 1 stops monitoring (expected behavior):**
   ```
   18:12:21 
   18:12:22 
   18:12:23 
   ```

5. **Terminal 3**: Attempt re-attach (FAILS)
   ```bash
   bpftime attach 608542
   ```
   
   **Re-attach log shows the problem:**
   ```
   [2025-07-09 18:12:31][info][609495] Agent already initialized, skipping re-initializing..
   ```
   
   **Terminal 1 shows no monitoring activity:**
   ```
   18:12:28 
   18:12:29 
   18:12:30 
   18:12:31
   ```

## Additional Testing

Even after killing the bpftime load process and restarting it, the problem persists:

1. Kill Terminal 1 process and restart `bpftime load ./example/malloc/malloc`
2. Attempt `bpftime attach 608542` again
3. **Same result**: "Agent already initialized, skipping re-initializing.." and no monitoring

**Fresh restart logs:**
```
[2025-07-09 18:13:29][info][610265] Initialize syscall server
[2025-07-09 18:13:29][info][610265] Global shm constructed. shm_open_type 0 for bpftime_maps_shm
[2025-07-09 18:13:29][info][610265] Global shm initialized
[2025-07-09 18:13:29][info][610265] bpftime-syscall-server started
[2025-07-09 18:13:29][info][610265] Created uprobe/uretprobe perf event handler, module name /lib/x86_64-linux-gnu/libc.so.6, offset 98860
```

**Still fails on attach:**
```
[2025-07-09 18:13:35][info][610350] Agent already initialized, skipping re-initializing..
```

## Root Cause Analysis

The issue appears to be that `bpftime detach` does not properly clean up the agent state within the target process. When a subsequent attach is attempted:

1. The attach logic detects that an agent was previously initialized in the target process
2. It skips re-initialization with the message "Agent already initialized, skipping re-initializing.."
3. However, the previously initialized agent is no longer functional after the detach
4. This results in a "zombie" state where the process appears attached but no monitoring occurs

## Expected Behavior

After `bpftime detach`, subsequent `bpftime attach` operations should:
1. Properly re-initialize the agent in the target process
2. Resume monitoring functionality 
3. Show the same initialization logs as the first attach

## Impact

This bug makes it impossible to:
- Re-attach to a process after detaching
- Perform iterative testing/debugging workflows
- Use attach/detach operations in automated testing or monitoring scenarios

## Workaround

Currently, the only workaround is to restart the target process entirely before attempting to re-attach.

## Additional Notes

- This issue is 100% reproducible with the provided steps
- The problem persists across bpftime process restarts
- Only restarting the target process resolves the issue
- No code modifications were made to the bpftime codebase

## Reproduction Results (2025-07-25)

Successfully reproduced the issue on the latest master branch.

### Test Commands Using Local Build Binaries

```bash
# Setup - Build everything first
cd /root/yunwei37/bpftime
make build
make -C example/malloc

# Terminal 1: Start bpftime load with local build binary
export BPFTIME_LOG_LEVEL=debug
./build/tools/cli/bpftime load ./example/malloc/malloc 2>&1 | tee /tmp/bpftime_load_debug.log

# Terminal 2: Start victim process and capture PID
cd /root/yunwei37/bpftime/example/malloc
export BPFTIME_LOG_LEVEL=debug
./victim 2>&1 | tee /tmp/victim_debug.log &
export VICTIM_PID=$!
echo "Victim PID: $VICTIM_PID"

# Terminal 3: Test attach/detach cycle with local build binary
cd /root/yunwei37/bpftime
export BPFTIME_LOG_LEVEL=debug

# First attach (should work)
./build/tools/cli/bpftime attach $VICTIM_PID
# Expected: "Initializing agent.." in logs
# Expected: Terminal 1 shows "pid=XXX malloc calls: YY"

# Detach
./build/tools/cli/bpftime detach
# Expected: "Detaching.." in victim logs
# Expected: Terminal 1 stops showing malloc calls

# Re-attach attempt (this will fail)
./build/tools/cli/bpftime attach $VICTIM_PID  
# BUG: Shows "Agent already initialized, skipping re-initializing.."
# BUG: No monitoring in Terminal 1

# Alternative: Test with specific PIDs
# Replace $VICTIM_PID with actual PID from ps aux | grep victim
```

### Automated Test Script

Create and run this test script:

```bash
#!/bin/bash
# Save as: /tmp/test_attach_detach.sh

# Configuration
BPFTIME_DIR="/root/yunwei37/bpftime"
BPFTIME_BIN="$BPFTIME_DIR/build/tools/cli/bpftime"
export BPFTIME_LOG_LEVEL=debug
export LD_LIBRARY_PATH=$BPFTIME_DIR/build/runtime:$LD_LIBRARY_PATH

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Building bpftime and malloc example..."
cd $BPFTIME_DIR
make build
make -C example/malloc

echo -e "\n${GREEN}Starting bpftime load...${NC}"
$BPFTIME_BIN load ./example/malloc/malloc 2>&1 | tee /tmp/bpftime_load.log &
LOAD_PID=$!
sleep 2

echo -e "\n${GREEN}Starting victim process...${NC}"
cd example/malloc
./victim 2>&1 | tee /tmp/victim.log &
VICTIM_PID=$!
echo "Victim PID: $VICTIM_PID"
sleep 2

echo -e "\n${GREEN}First attach (should work)...${NC}"
$BPFTIME_BIN attach $VICTIM_PID
sleep 3

# Check if monitoring is working
if grep -q "malloc calls:" /tmp/bpftime_load.log; then
    echo -e "${GREEN}✓ First attach successful - monitoring active${NC}"
else
    echo -e "${RED}✗ First attach failed - no monitoring${NC}"
fi

echo -e "\n${GREEN}Detaching...${NC}"
$BPFTIME_BIN detach
sleep 2

# Clear log to check for new monitoring
> /tmp/bpftime_load.log

echo -e "\n${GREEN}Re-attaching (testing bug)...${NC}"
$BPFTIME_BIN attach $VICTIM_PID
sleep 3

# Check if re-attach worked
if grep -q "malloc calls:" /tmp/bpftime_load.log; then
    echo -e "${GREEN}✓ Re-attach successful - monitoring active${NC}"
else
    echo -e "${RED}✗ Re-attach failed - BUG CONFIRMED${NC}"
    grep "Agent already initialized" /tmp/victim.log
fi

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
kill $VICTIM_PID 2>/dev/null
kill $LOAD_PID 2>/dev/null

echo -e "\nLogs saved to:"
echo "  - /tmp/bpftime_load.log"
echo "  - /tmp/victim.log"
```

### Quick Test Commands

```bash
# One-liner to test with local build binary
cd /root/yunwei37/bpftime && make build && ./build/tools/cli/bpftime --version

# Test if agent library exists in build directory
ls -la ./build/runtime/libbpftime-agent.so

# Check if victim is using the library after attach
ps aux | grep victim | grep -v grep | awk '{print $2}' | xargs -I{} cat /proc/{}/maps | grep bpftime-agent

# Monitor agent initialization in real-time
tail -f /tmp/victim_debug.log | grep -E "(Initializing agent|Agent already initialized)"

# Test with specific debug output for state tracking
cd /root/yunwei37/bpftime
BPFTIME_LOG_LEVEL=debug SPDLOG_LEVEL=debug ./build/tools/cli/bpftime attach $(pgrep victim)

# Check agent library path used by bpftime CLI
ldd ./build/tools/cli/bpftime | grep agent

# Set library path if needed
export LD_LIBRARY_PATH=/root/yunwei37/bpftime/build/runtime:$LD_LIBRARY_PATH
```

### Reproduction Steps Used

```bash
# 1. Build the malloc example
cd /root/yunwei37/bpftime
make -C example/malloc

# 2. Start bpftime load with debug logging (using build binary)
BPFTIME_LOG_LEVEL=debug ./build/tools/cli/bpftime load ./example/malloc/malloc 2>&1 | tee /tmp/bpftime_load_debug.log &

# 3. Start victim process
cd example/malloc
BPFTIME_LOG_LEVEL=debug ./victim 2>&1 | tee /tmp/victim_debug.log &
# Note actual PID: 320389

# 4. First attach (works correctly)
cd /root/yunwei37/bpftime
BPFTIME_LOG_LEVEL=debug ./build/tools/cli/bpftime attach 320389
# Log shows: "Initializing agent.."
# Monitoring works: "pid=320389 malloc calls: 20"

# 5. Detach
BPFTIME_LOG_LEVEL=debug ./build/tools/cli/bpftime detach
# Monitoring stops as expected

# 6. Re-attach attempt (FAILS)
BPFTIME_LOG_LEVEL=debug ./build/tools/cli/bpftime attach 320389
# Log shows: "Agent already initialized, skipping re-initializing.."
# NO monitoring occurs despite successful injection
```

### Key findings:

1. **First attach (PID 320389)**: Works correctly
   - Shows "Initializing agent.." in logs
   - Monitoring output appears: `pid=320389 malloc calls: 20`

2. **Detach**: Successfully stops monitoring
   - Sends SIGUSR1 to process
   - Monitoring output stops appearing

3. **Re-attach attempt**: FAILS as described
   - Shows "Agent already initialized, skipping re-initializing.." in victim process logs
   - NO monitoring output appears despite successful injection message
   - The agent is in a "zombie" state - appears attached but not functional

The issue is confirmed: `bpftime detach` does not properly clean up the agent state, preventing successful re-attachment.

## Root Cause Analysis

After investigating the code, the root cause has been identified:

### The Problem

1. **Agent Initialization** (`runtime/agent/agent.cpp:134-142`):
   ```cpp
   static int initialized = 0;
   
   int expected = 0;
   if (!__atomic_compare_exchange_n(&initialized, &expected, 1,
                                   false, __ATOMIC_SEQ_CST,
                                   __ATOMIC_SEQ_CST)) {
       SPDLOG_INFO("Agent already initialized, skipping re-initializing..");
       return;
   }
   ```
   - Uses a static variable `initialized` to prevent multiple initializations
   - Sets it from 0 to 1 atomically during first initialization

2. **Detach Handler** (`runtime/agent/agent.cpp:107-118`):
   ```cpp
   static void sig_handler_sigusr1(int sig)
   {
       SPDLOG_INFO("Detaching..");
       if (int err = ctx_holder.ctx.destroy_all_attach_links(); err < 0) {
           SPDLOG_ERROR("Unable to detach: {}", err);
           return;
       }
       shm_holder.global_shared_memory.remove_pid_from_alive_agent_set(getpid());
       SPDLOG_DEBUG("Detaching done");
       bpftime_logger_flush();
   }
   ```
   - Destroys all attach links
   - Removes PID from alive agent set
   - **BUT does NOT reset the `initialized` variable back to 0**

3. **Detach Command** (`tools/cli/main.cpp:334`):
   - Sends SIGUSR1 to all processes in the alive agent set
   - The signal triggers `sig_handler_sigusr1` in the target process

### The Bug

The `initialized` static variable persists with value 1 after detach, causing any subsequent attach attempt to skip initialization with "Agent already initialized, skipping re-initializing.." message.

### Solution

The fix would be to reset the `initialized` variable to 0 in the `sig_handler_sigusr1` function after successful detachment:

```cpp
static void sig_handler_sigusr1(int sig)
{
    SPDLOG_INFO("Detaching..");
    if (int err = ctx_holder.ctx.destroy_all_attach_links(); err < 0) {
        SPDLOG_ERROR("Unable to detach: {}", err);
        return;
    }
    shm_holder.global_shared_memory.remove_pid_from_alive_agent_set(getpid());
    
    // Reset initialization state to allow re-attachment
    __atomic_store_n(&initialized, 0, __ATOMIC_SEQ_CST);
    
    SPDLOG_DEBUG("Detaching done");
    bpftime_logger_flush();
}
```

## Testing Results

After applying the fix and rebuilding, testing shows that the simple atomic store reset is not sufficient to solve the issue. The problem appears to be more complex:

1. **The fix was applied**: Added `__atomic_store_n(&initialized, 0, __ATOMIC_SEQ_CST);` to the detach signal handler
2. **Code was rebuilt**: Confirmed the fix is in the compiled library
3. **Issue persists**: Re-attach still fails silently - no monitoring resumes after detach

### Observations

- First attach works correctly with monitoring
- Detach stops monitoring as expected  
- Re-attach appears successful from CLI but monitoring does not resume
- No "Agent already initialized" message appears in logs with the fix
- The agent seems to be in a non-functional state after re-attach

### Possible Root Causes

1. **Frida Injection State**: The agent is injected via Frida, and the injection state may not be fully cleaned up
2. **Handler Context**: The `ctx_holder` object may need proper re-initialization beyond just resetting the flag
3. **Shared Memory State**: The shared memory connections may need to be re-established
4. **Signal Handler Limitations**: The detach handler runs in signal context with limited capabilities

### Next Steps

A more comprehensive fix is needed that:
- Properly cleans up all agent state during detach
- Ensures the agent can be fully re-initialized on re-attach
- Handles the Frida injection lifecycle correctly

The issue requires deeper investigation into the agent lifecycle and Frida injection mechanism.

## Detailed Analysis of Attach/Detach Process

### Current Attach Process

1. **CLI Command**: `bpftime attach <PID>`
   - Calls `inject_by_frida(pid, agent_path, "")` 
   - Uses Frida to inject `libbpftime-agent.so` into the target process
   - Calls the `bpftime_agent_main` function as entry point

2. **Agent Initialization**: 
   - Checks if already initialized using static `initialized` variable
   - If already initialized, skips with "Agent already initialized" message
   - Otherwise:
     - Registers signal handlers (SIGUSR1 for detach)
     - Initializes shared memory
     - Sets up attach implementations (uprobe, syscall trace, etc.)
     - Calls `init_attach_ctx_from_handlers()` to load eBPF programs
     - Adds PID to alive agent set in shared memory

### Current Detach Process  

1. **CLI Command**: `bpftime detach`
   - Iterates through all PIDs in alive agent set
   - Sends SIGUSR1 to each PID

2. **Agent Signal Handler** (`sig_handler_sigusr1`):
   - Calls `ctx.destroy_all_attach_links()` - unhooks all probes
   - Removes PID from alive agent set
   - **BUT**: The agent library remains loaded in the process!

### The Core Problem

When re-attaching:
1. Frida re-injects the library and calls `bpftime_agent_main` again
2. The static `initialized` variable is still 1 (because the library was never unloaded)
3. The function returns early without re-initializing
4. No eBPF programs are loaded, no probes are attached

### Proposed Solution: Signal-Based Re-initialization

Instead of trying to re-inject with Frida, we should:

1. **On Attach**: Check if the agent is already loaded
   - Option A: Check if PID was previously in alive set but removed (detached)
   - Option B: Try sending a different signal (e.g., SIGUSR2) for re-attach
   - Option C: Check if the library is already loaded in `/proc/<pid>/maps`

2. **New Re-attach Flow**:
   ```
   if (agent_already_loaded(pid)) {
       // Send SIGUSR2 to trigger re-initialization
       kill(pid, SIGUSR2);
   } else {
       // First time attach - inject with Frida
       inject_by_frida(pid, agent_path, "");
   }
   ```

3. **Agent Changes**:
   - Add SIGUSR2 handler for re-initialization
   - The handler would:
     - Reset the `initialized` flag
     - Call `bpftime_agent_main` again to reload programs
     - Or directly call `init_attach_ctx_from_handlers()`

### Alternative Solutions

1. **Unload Library on Detach**: 
   - More complex - would need to ensure all hooks are removed
   - Might cause stability issues

2. **Separate Init State**: 
   - Keep `initialized` for one-time setup (signal handlers, etc.)
   - Add `attached` flag for eBPF program state
   - Allow re-attachment by just reloading programs

3. **Persistent Agent Mode**:
   - Agent stays loaded and ready
   - Detach only removes probes, not the agent
   - Re-attach just reloads the programs

The signal-based approach is cleanest and avoids the complexity of multiple Frida injections.