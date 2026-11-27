# bpftime macOS Virtualization.framework launcher

Rootless ARM64 Linux VM tooling that lets unmodified Linux eBPF control-plane
binaries (`bpftrace`, BCC tools, `libbpf` programs, the OpenTelemetry eBPF
profiler) run on Apple Silicon macOS against native macOS processes.

The VM runs an Ubuntu 24.04 arm64 cloud image under the built-in
[Virtualization.framework](https://developer.apple.com/documentation/virtualization):
NAT virtio networking, a virtiofs share of host `/Users` mounted at the same
guest path, SSH key access generated on `setup`, and EFI boot with NVRAM
recreated on every run. All VM state lives under
`$HOME/.bpftime/macos-vm` (override with `BPFTIME_VM_DIR`).

## Files

| File | Role |
| --- | --- |
| `bpftime_vz.swift` | `bpftime-vz` VM manager: `setup` / `run` / `ip` |
| `qcow2_to_raw.py` | One-time QCOW2 → raw converter (stock macOS has no `qemu-img` and Ubuntu publishes no raw arm64 image) |
| `bpftime_macos_exec.c` | Linux-guest `binfmt_misc` interpreter for Mach-O binaries; forwards exec to the Darwin host over the bridge |
| `register_binfmt.sh` | Registers Mach-O magic bytes with `binfmt_misc` inside the VM |
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
a separate terminal. Useful flags: `--vm-dir DIR`, `--memory-gb N`,
`--cpus N` (defaults: 8 GiB, host logical CPU count).

## Run eBPF tools with one command

The wrapper syncs the bpftime source into the VM on first use, builds the
syscall-server and the Mach-O exec interpreter there, mounts virtiofs and
registers `binfmt_misc`, starts `bpftime-darwin-host` on the Mac, then execs
the requested tool:

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
(host build tree containing `bpftime-darwin-host` and `libbpftime-agent.dylib`),
`BPFTIME_DARWIN_PORT` (default 49512), `BPFTIME_MACOS_SUDO=1|0|auto`.

## OpenTelemetry eBPF profiler

The end-to-end cross-end runner (Linux collector inside the VM, native Mach-O
victim on the host) lives at
[`example/otel-ebpf-profiler/scripts/run-macos.sh`](../../example/otel-ebpf-profiler/scripts/run-macos.sh)
and takes `OTEL_COLLECTOR_BIN` plus the same `BPFTIME_VM_DIR` /
`BPFTIME_BUILD_DIR` overrides.

## Companion components

This directory contains only the VM launcher and host-side wrapper. The
Darwin agent (`libbpftime-agent.dylib`), `bpftime-darwin-host`, and the
syscall-server Mach-O bridge are delivered by companion pull requests;
`bpftime-macos` reports a clear build-artifact error until those are built.
