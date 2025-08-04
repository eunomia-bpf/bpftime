# State Tracking Table

## Process State Matrix

| Agent Loaded | In Alive Set | Initialized | Attached | State Description | CLI Action |
|--------------|--------------|-------------|----------|-------------------|------------|
| No | No | - | - | Clean Process | inject_by_frida() |
| Yes | No | 1 | 0 | Detached | Send SIGUSR2 |
| Yes | Yes | 1 | 1 | Attached & Active | Nothing (already attached) |
| Yes | No | 1 | 1 | Error: Stale State | Report error, suggest restart |
| Yes | No | 0 | - | Error: Corrupt State | Report error, suggest restart |

## State Detection Methods

### 1. Check Agent Loaded
```bash
grep -q "libbpftime-agent" /proc/$PID/maps
```

### 2. Check In Alive Set
```cpp
bpftime::shm_holder.global_shared_memory.is_pid_in_alive_agent_set(pid)
```

### 3. Check Process Exists
```bash
kill -0 $PID 2>/dev/null
```

## CLI Decision Tree

```
if (!process_exists(pid))
    → Error: Process not found

else if (!agent_loaded(pid))
    → First attach: inject_by_frida()

else if (in_alive_set(pid))
    → Info: Already attached

else
    → Re-attach: kill(pid, SIGUSR2)
```

## Signal Usage

| Signal | Sender | Handler Function | Purpose |
|--------|--------|------------------|---------|
| SIGUSR1 | CLI (detach) | sig_handler_sigusr1 | Detach: Remove probes, clear attached state |
| SIGUSR2 | CLI (re-attach) | sig_handler_sigusr2 | Re-attach: Reload eBPF programs |

## Shared Memory State

The shared memory tracks:
- `alive_agent_set`: Set of PIDs with active agents
- `handler_infos`: eBPF program information
- `map_infos`: eBPF map information

## Process Lifecycle Examples

### Example 1: Normal Attach → Detach → Re-attach
```
1. Process 1234 starts (clean)
2. bpftime attach 1234
   - inject_by_frida() → agent loaded
   - initialized=1, attached=1
   - PID added to alive_agent_set
3. bpftime detach
   - SIGUSR1 → probes removed
   - attached=0
   - PID removed from alive_agent_set
4. bpftime attach 1234
   - Agent detected in /proc/1234/maps
   - PID not in alive_agent_set
   - SIGUSR2 → reload programs
   - attached=1
   - PID added back to alive_agent_set
```

### Example 2: Process Crash During Attached State
```
1. Process 1234 attached and running
2. Process crashes (agent still in memory maps)
3. bpftime attach 1234
   - Agent detected but PID not in alive_agent_set
   - Stale state detected
   - Error reported to user
```

### Example 3: Double Attach Attempt
```
1. Process 1234 already attached
2. bpftime attach 1234
   - Agent detected in maps
   - PID found in alive_agent_set
   - Info: "Already attached"
   - No action taken
```