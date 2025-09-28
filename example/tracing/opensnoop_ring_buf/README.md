# opensnoop

Trace file open or close syscalls in a process.

Since the program is not attached to all syscall events, it is not necessary to filter the events for pid and uid in the program. It can also speedup the program and reduce the overhead.

## Usage

```console
$ sudo ~/.bpftime/bpftime load ./example/opensnoop/opensnoop
[2023-10-09 04:36:33.891] [info] manager constructed
[2023-10-09 04:36:33.892] [info] global_shm_open_type 0 for bpftime_maps_shm
[2023-10-09 04:36:33][info][23999] Enabling helper groups ffi, kernel, shm_map by default
PID    COMM              FD ERR PATH
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
72101  victim             3   0 test.txt
```

In another terminal, run the victim program:

```console
$ sudo ~/.bpftime/bpftime start -s example/opensnoop/victim
[2023-10-09 04:38:16.196] [info] Entering new main..
[2023-10-09 04:38:16.197] [info] Using agent /root/.bpftime/libbpftime-agent.so
[2023-10-09 04:38:16.198] [info] Page zero setted up..
[2023-10-09 04:38:16.198] [info] Rewriting executable segments..
[2023-10-09 04:38:19.260] [info] Loading dynamic library..
...
test.txt closed
Opening test.txt
test.txt opened, fd=3
Closing test.txt...
```
