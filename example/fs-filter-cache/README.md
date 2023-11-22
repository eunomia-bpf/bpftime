# fs-filter-cache

Trace file open or close syscalls in a process.

Since the program is not attached to all syscall events, it is not necessary to filter the events for pid and uid in the program. It can also speedup the program and reduce the overhead.

## Usage

run agent

```console
# AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so find ./example
```
