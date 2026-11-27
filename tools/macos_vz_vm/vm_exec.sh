#!/bin/bash
# SPDX-License-Identifier: MIT
#
# vm_exec.sh - run a command inside the bpftime-vz ARM64 Ubuntu VM over SSH
# without any interactive prompt. The VM IP, gateway and SSH key path are
# recorded by `bpftime-vz run` in vm-info.json under the VM state directory.
#
# Usage:
#   vm_exec.sh [--vm-dir DIR] <remote command args...>
#   vm_exec.sh [--vm-dir DIR] --wait-ssh [timeout-seconds]
#
# Environment overrides:
#   BPFTIME_VM_DIR   VM state directory (default: $HOME/.bpftime/macos-vm)

set -euo pipefail

VM_DIR="${BPFTIME_VM_DIR:-$HOME/.bpftime/macos-vm}"

if [[ "${1:-}" == "--vm-dir" ]]; then
	VM_DIR="$2"
	shift 2
fi

INFO="$VM_DIR/vm-info.json"

read_json_field() {
	# Minimal plutil-free JSON extraction (macOS/Linux compatible).
	local key="$1"
	python3 - "$INFO" "$key" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)[sys.argv[2]])
PY
}

wait_ssh() {
	local timeout="${1:-180}"
	local deadline=$(( $(date +%s) + timeout ))
	until [[ -f "$INFO" ]]; do
		if (( $(date +%s) > deadline )); then
			echo "vm_exec: timed out waiting for $INFO" >&2
			return 1
		fi
		sleep 2
	done
	local ip key
	ip="$(read_json_field vm_ip)"
	key="$(read_json_field ssh_key)"
	chmod 0600 "$key" 2>/dev/null || true
	until ssh -i "$key" \
		-o BatchMode=yes \
		-o StrictHostKeyChecking=no \
		-o UserKnownHostsFile=/dev/null \
		-o ConnectTimeout=3 \
		"ubuntu@$ip" true 2>/dev/null; do
		if (( $(date +%s) > deadline )); then
			echo "vm_exec: SSH to ubuntu@$ip did not become ready" >&2
			return 1
		fi
		sleep 2
	done
	echo "vm_exec: SSH ready at ubuntu@$ip"
}

if [[ "${1:-}" == "--wait-ssh" ]]; then
	wait_ssh "${2:-180}"
	exit 0
fi

IP="$(read_json_field vm_ip)"
SSH_KEY="$(read_json_field ssh_key)"
SSH_USER="$(read_json_field ssh_user)"

chmod 0600 "$SSH_KEY" 2>/dev/null || true
remote_cmd=""
for a in "$@"; do
	# Re-quote each argument so SSH's remote shell rebuilds the exact argv
	# array instead of re-splitting spaces and shell metacharacters.
	remote_cmd+=" $(printf '%q' "$a")"
done
exec ssh -i "$SSH_KEY" \
	-o BatchMode=yes \
	-o StrictHostKeyChecking=no \
	-o UserKnownHostsFile=/dev/null \
	-o ConnectTimeout=10 \
	"${SSH_USER}@${IP}" "$remote_cmd"
