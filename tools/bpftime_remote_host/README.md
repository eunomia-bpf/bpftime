# bpftime-remote-host

Cross-platform target daemon for bpftime's generic cross-machine control
plane. It runs on a remote **Linux** or **macOS** machine, owns the local
bpftime shared-memory segment, and serves the frame protocol over TCP so a
Linux control plane can drive tracing on either OS while the data plane
(agent + Boost SHM + JIT + maps/ringbuf) stays fully local to the target.

## Wire protocol

Each frame is an 8-byte little-endian header (`uint32_t json_len`,
`uint32_t blob_len`) followed by a UTF-8 JSON document and an optional binary
blob. Operations include `hello`, `proc_check`, `proc_maps`, `fetch_elf`
(alias `synth_elf`), the `map_*` family, `prog_load`, `perf_event_open`,
`bpf_link_create`, `perf_event_enable`/`perf_event_disable`,
`spawn_cmd`/`resume_cmd`, and the persistent `subscribe_events` stream
(bounded to 64 events per poll slice).

## Security

- Binds `127.0.0.1` by default. Binding `0.0.0.0` is rejected unless
  `--allow-any` is passed.
- A shared secret is mandatory on the **first frame of every connection**
  (`hello` for RPC sockets, `subscribe_events` for event sockets), verified
  with a constant-time comparison. The token comes from `BPFTIME_REMOTE_TOKEN`
  or `--token-file`; if neither is set, 32 random bytes (64 hex chars) are
  generated into `$HOME/.bpftime/remote.token` with `0600` permissions.
- Any frame on an unauthenticated socket is answered with
  `{"ok":false,"error":"unauthorized"}` and the connection is closed.

## Platform backends

- **Linux** (`linux_target.cpp`): real `/proc/<pid>/{maps,comm,status}` and
  `readlink(/proc/<pid>/exe)`, on-disk ELF fetch with magic validation, and
  `LD_PRELOAD` spawn that raises `SIGSTOP` before `execve` when held, resumed
  with `SIGCONT`. Incremental attach uses the agent IPC abstract socket.
- **macOS** (`darwin_target.cpp`, `macho_reader.cpp`, `elf_synth.cpp`):
  synthesizes `/proc/<pid>/maps` and 64-bit stub ELFs from Mach-O / Fat /
  `dyld_shared_cache` images (DOF USDT -> `.note.stapsdt`, ARM64
  `.eh_frame`), and spawns with `POSIX_SPAWN_START_SUSPENDED` + `SIGCONT`.

## Usage

```
bpftime-remote-host [-p|--port 49512] [--bind 127.0.0.1]
                    [--token-file PATH] [--allow-any]
```

The end-to-end test `test/e2e_remote_host.py` launches the daemon and
exercises authentication, ELF fetch, map CRUD, held spawn/resume, and (where
an agent shared library is available) the uprobe -> ringbuf -> TCP event path
on both operating systems.
