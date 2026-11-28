#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Register bpftime-remote-exec as the binfmt_misc interpreter for Mach-O 64
# and fat universal binaries inside the bpftime ARM64 Linux VM. After this,
# execve() of any /Users/... Mach-O program (e.g. bpftrace -c ./app) is
# transparently serviced by the remote target through the BPFTIME_REMOTE_HOST
# bridge. bpftime-remote-exec is built by the generic remote bridge component
# (tools/bpftime_remote), not by this VM launcher.
#
# Run inside the VM (cloud-init grants the ubuntu user passwordless sudo):
#   sudo ./register_binfmt.sh [path-to-bpftime-remote-exec]

set -euo pipefail

INTERP="${1:-/usr/local/bin/bpftime-remote-exec}"

if [[ ! -x "$INTERP" ]]; then
	echo "register_binfmt: interpreter not found or not executable: $INTERP" >&2
	exit 1
fi

# binfmt_misc must be mounted (present on every Ubuntu cloud kernel).
sudo mkdir -p /proc/sys/fs/binfmt_misc
sudo mountpoint -q /proc/sys/fs/binfmt_misc || \
	sudo mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc

# Mach-O 64-bit MH_MAGIC_64 = 0xFEEDFACF, stored little-endian, so the
# on-disk byte sequence is CF FA ED FE (not CF FA FE CF). binfmt_misc
# compares raw file bytes, so the order must match exactly. printf (not
# echo) guarantees the \x escapes are decoded under any shell.
printf ':bpftime_macho64:M::\xcf\xfa\xed\xfe::%s:F\n' "$INTERP" | \
	sudo tee /proc/sys/fs/binfmt_misc/register >/dev/null

# Fat/universal binary FAT_MAGIC = 0xCAFEBABE, big-endian on disk:
# CA FE BA BE.
printf ':bpftime_fat:M::\xca\xfe\xba\xbe::%s:F\n' "$INTERP" | \
	sudo tee /proc/sys/fs/binfmt_misc/register >/dev/null

echo "binfmt_misc registrations:"
cat /proc/sys/fs/binfmt_misc/bpftime_macho64
echo
cat /proc/sys/fs/binfmt_misc/bpftime_fat
