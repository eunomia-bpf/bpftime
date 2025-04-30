# syscall micro-benchmark

## userspace syscall

### run

```sh
sudo ~/.bpftime/bpftime load benchmark/syscall/syscall
# or
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/syscall/syscall
```

in another shell, run the target program with eBPF inside:

```sh
sudo ~/.bpftime/bpftime start -s benchmark/syscall/victim
# or
AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/attach/text_segment_transformer/libbpftime-agent-transformer.so benchmark/syscall/victim
```

## results (2023)

Tested on `6.2.0-32-generic` kernel and `Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz`.

### baseline

Average time usage 213.62178ns,  count 1000000

### userspace syscall

Average time usage 446.19869ns,  count 1000000

### kernel tracepoint

Average time usage 365.44980ns,  count 1000000
