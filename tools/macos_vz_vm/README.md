# bpftime macOS Virtualization.framework launcher

Rootless ARM64 Linux VM tooling that lets unmodified Linux eBPF control-plane
binaries (`bpftrace`, BCC tools, `libbpf` programs, the OpenTelemetry eBPF
profiler) run on Apple Silicon macOS against native macOS processes.

The VM runs an Ubuntu 24.04 arm64 cloud image under the built-in
[Virtualization.framework](https://developer.apple.com/documentation/virtualization):
NAT virtio networking, a virtiofs share of host `/Users` mounted at the same
guest path, SSH key access generated on `setup`, direct kernel boot when a
kernel/initrd pair is present (otherwise EFI boot with NVRAM recreated on every
run), and an exclusive `flock` on `disk.raw.lock` that serializes concurrent
runs. All VM state lives under `$HOME/.bpftime/macos-vm` (override with
`BPFTIME_VM_DIR`).

This directory is a thin, macOS-specific convenience layer on top of the
generic, platform-neutral `BPFTIME_REMOTE_HOST` remote bridge: the
`bpftime-remote-host` target daemon, the syscall-server `remote_bridge`
client, and the `bpftime-remote-exec` workload spawner are delivered by
companion pull requests and work for both remote Linux and remote macOS
targets. No wire protocol or bridge client code lives here.

## Files

| File | Role |
| --- | --- |
| `bpftime_vz.swift` | `bpftime-vz` VM manager: `setup` / `run` / `ip` |
| `qcow2_to_raw.py` | One-time QCOW2 → raw converter (stock macOS has no `qemu-img` and Ubuntu publishes no raw arm64 image) |
| `register_binfmt.sh` | Registers Mach-O magic bytes with `binfmt_misc` inside the VM, using the generic `bpftime-remote-exec` interpreter |
| `bpftime-macos` | One-command host wrapper: prepares the VM and execs a Linux tool in it |
| `vm_exec.sh` | Small SSH helper for running commands in the VM |
| `entitlements.plist` | `com.apple.security.virtualization` entitlement for codesigning |

## Build the launcher (Apple Silicon host)

```bash
cd tools/macos_vz_vm
/usr/bin/swiftc -O bpftime_vz.swift -o bpftime-vz -framework Virtualization
codesign --sign - --entitlements entitlements.plist --force bpftime-vz
```

## Start the VM

```bash
./bpftime-vz setup          # download image (checksum-verified), SSH key, cloud-init seed
./bpftime-vz run            # boot the VM; writes vm-info.json once SSH is reachable
./bpftime-vz ip             # re-discover the guest IP if discovery timed out
```

`bpftime-macos` never starts or stops the VM; keep `bpftime-vz run` alive in
a separate terminal. A second concurrent `bpftime-vz run` fails immediately
because it cannot acquire the exclusive `disk.raw.lock`. Useful flags:
`--vm-dir DIR`, `--memory-gb N`, `--cpus N` (defaults: 8 GiB, host logical
CPU count).

### Direct kernel boot (optional)

Place `vmlinuz` and `initrd.img` in the VM directory and `bpftime-vz run`
uses `VZLinuxBootLoader` with
`console=hvc0 root=LABEL=cloudimg-rootfs ro fsck.mode=force fsck.repair=yes`
instead of EFI boot. Without both files it recreates a clean
`VZEFIVariableStore` and boots the image's removable EFI loader.

## Run eBPF tools with one command

The wrapper syncs the bpftime source into the VM on first use, installs the
build prerequisites, builds the syscall-server and `bpftime-remote-exec`
there, mounts virtiofs and registers `binfmt_misc`, starts
`bpftime-remote-host` on the Mac bound to the VZ NAT gateway with a
session token, then execs the requested tool with `BPFTIME_REMOTE_HOST` and
`BPFTIME_REMOTE_TOKEN` exported (the target OS is negotiated automatically by
the bridge's authenticated `hello` handshake):

```bash
# bpftrace uprobe against a native macOS Mach-O workload
tools/macos_vz_vm/bpftime-macos bpftrace -e 'uprobe:libc:malloc { @[ustack] = count(); }' \
    -c /Users/me/victim

# BCC / python tools
tools/macos_vz_vm/bpftime-macos execsnoop
tools/macos_vz_vm/bpftime-macos python3 trace.py

# force a rebuild of the VM-side bpftime components
tools/macos_vz_vm/bpftime-macos --rebuild bpftrace --version
```

Environment overrides: `BPFTIME_VM_DIR` (VM state), `BPFTIME_BUILD_DIR`
(host build tree containing `bpftime-remote-host` and
`libbpftime-agent.dylib`), `BPFTIME_REMOTE_PORT` (default 49512),
`BPFTIME_MACOS_SUDO=1|0|auto`.

## Security: gateway-only bind and session token

The host daemon can spawn programs and load BPF objects on the Mac, so its
listener must not be reachable from the wider network. Both `bpftime-macos`
and the OpenTelemetry runner enforce the access-control contract recorded in
the maintainer decision (#1014):

- `bpftime-remote-host` is started with `--bind <gateway-ip>`, i.e. the VZ NAT
  gateway address from `vm-info.json` (for example `192.168.64.1`), never
  `0.0.0.0`. Only the local VM has a route to that interface; other hosts on
  the LAN cannot connect.
- A 256-bit session token (32 random bytes rendered as 64 hex characters) is
  generated with `0600` permissions into the wrapper state directory
  (`${TMPDIR:-/tmp}/bpftime-macos/remote.token`, reused across invocations) or
  a per-run token file for the OpenTelemetry runner, and passed with
  `--token-file`. The daemon demands this token in the very first frame
  (`hello` / `subscribe_events`) of every TCP connection and rejects any
  unauthenticated socket.
- The same secret is exported to the VM as `BPFTIME_REMOTE_TOKEN`, so the
  syscall-server `remote_bridge` client and `bpftime-remote-exec` authenticate
  automatically. No token is placed on the kernel command line or in
  world-readable files.

## OpenTelemetry eBPF profiler

The end-to-end cross-end runner (Linux collector inside the VM, native Mach-O
victim on the host) lives at
[`example/otel-ebpf-profiler/scripts/run-macos.sh`](../../example/otel-ebpf-profiler/scripts/run-macos.sh)
and takes `OTEL_COLLECTOR_BIN` plus the same `BPFTIME_VM_DIR` /
`BPFTIME_BUILD_DIR` overrides.

## Companion components

This directory contains only the VM launcher and host-side wrapper. The
Darwin agent (`libbpftime-agent.dylib`), the cross-platform
`bpftime-remote-host` daemon, the syscall-server `remote_bridge` client, and
`bpftime-remote-exec` are delivered by companion pull requests;
`bpftime-macos` reports a clear build-artifact error until those are built.
