#!/usr/bin/env bash
# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#
# End-to-end rootless OpenTelemetry eBPF profiler on Apple Silicon macOS.
#
# Topology:
#   ARM64 Linux VM (bpftime-vz)                    macOS host (native arm64)
#   --------------------------------------------   ----------------------------------------
#   otelcol-ebpf-profiler (unmodified ELF arm64)   bpftime-darwin-host (:PORT, Boost SHM)
#     LD_PRELOAD=libbpftime-syscall-server.so  <->   synthesizes Mach-O/dyld stub ELFs,
#     BPFTIME_GO_SECCOMP=1                           translates libc:{malloc,free} -> xzone
#     BPFTIME_DARWIN_HOST=192.168.64.1:PORT        victim-macos (native Mach-O arm64)
#     BPFTIME_TARGET_OSX=1                           DYLD_INSERT_LIBRARIES=libbpftime-agent.dylib

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${BPFTIME_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MACOS_BUILD_DIR="${BPFTIME_BUILD_DIR:-${REPO_ROOT}/build}"
DARWIN_HOST_BIN="${DARWIN_HOST_BIN:-${MACOS_BUILD_DIR}/tools/bpftime_darwin_host/bpftime-darwin-host}"
AGENT_DYLIB="${DARWIN_AGENT_DYLIB:-${MACOS_BUILD_DIR}/runtime/agent/libbpftime-agent.dylib}"

VM_DIR="${BPFTIME_VM_DIR:-$HOME/.bpftime/macos-vm}"
if [[ ! -f "${VM_DIR}/vm-info.json" ]]; then
	echo "set BPFTIME_VM_DIR to a running bpftime-vz directory containing vm-info.json" >&2
	exit 1
fi

json_field() {
	python3 - "${VM_DIR}/vm-info.json" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
}

VM_IP="$(json_field vm_ip)"
GW_IP="$(json_field gateway_ip)"
SSH_KEY="$(json_field ssh_key)"
SSH_USER="$(json_field ssh_user)"

VM_BUILD_DIR="${VM_BUILD_DIR:-/home/${SSH_USER}/bpftime-build}"
VM_SERVER_SO="${VM_SERVER_SO:-${VM_BUILD_DIR}/runtime/syscall-server/libbpftime-syscall-server.so}"
COLLECTOR="${OTEL_COLLECTOR_BIN:-}"
if [[ -z "${COLLECTOR}" ]]; then
	echo "set OTEL_COLLECTOR_BIN to the dynamic arm64 Linux otelcol-ebpf-profiler binary (visible in VM)" >&2
	exit 1
fi

if [[ -z "${BPFTIME_DARWIN_PORT:-}" ]]; then
	PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
else
	PORT="${BPFTIME_DARWIN_PORT}"
fi
RUN_SECONDS="${RUN_SECONDS:-14}"
STARTUP_WAIT="${STARTUP_WAIT:-30}"
FLUSH_WAIT="${FLUSH_WAIT:-6}"
CONFIG="${OTEL_CONFIG:-${EXAMPLE_DIR}/config/otelcol.yaml}"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/macos-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
VICTIM_MACOS="${LOG_DIR}/victim-macos"

for f in "${DARWIN_HOST_BIN}" "${AGENT_DYLIB}" "${CONFIG}"; do
	if [[ ! -e "${f}" ]]; then
		echo "required host artifact not found: ${f}" >&2
		exit 1
	fi
done

cc -O2 -g -fno-builtin -Wall -Wextra "${EXAMPLE_DIR}/victim-src/victim.c" -o "${VICTIM_MACOS}"

SSH_OPTS=(-i "${SSH_KEY}" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8)
vssh() { ssh "${SSH_OPTS[@]}" "${SSH_USER}@${VM_IP}" "$@"; }

HOST_PID=""
HOST_SHM=""
VM_SHM=""
VM_COLLECTOR_LAUNCH_PID=""
VICTIM_PID=""

cleanup() {
	local status=$?
	trap - EXIT INT TERM
	[[ -n "${VICTIM_PID}" ]] && kill -9 "${VICTIM_PID}" 2>/dev/null || true
	vssh "rm -f /dev/shm/${VM_SHM:-bpftime_otel_vm_none}* 2>/dev/null || true" 2>/dev/null || true
	vssh "pkill -9 -f '[o]telcol-ebpf-profiler' 2>/dev/null || true" 2>/dev/null || true
	[[ -n "${VM_COLLECTOR_LAUNCH_PID}" ]] && kill -9 "${VM_COLLECTOR_LAUNCH_PID}" 2>/dev/null || true
	[[ -n "${HOST_PID}" ]] && kill -9 "${HOST_PID}" 2>/dev/null || true
	wait 2>/dev/null || true
	echo "logs: ${LOG_DIR}" >&2
	exit "${status}"
}
trap cleanup EXIT INT TERM

# 1. Start dedicated bpftime-darwin-host on macOS.
BPFTIME_AGENT_DYLIB="${AGENT_DYLIB}" \
BPFTIME_SHM_MEMORY_MB=1024 \
BPFTIME_LOG_OUTPUT="${LOG_DIR}/bpftime-host-shm.log" \
SPDLOG_LEVEL=info \
	"${DARWIN_HOST_BIN}" -p "${PORT}" >"${LOG_DIR}/darwin-host.log" 2>&1 &
HOST_PID=$!

HOST_SHM=""
for ((i = 0; i < 20; i++)); do
	if ! kill -0 "${HOST_PID}" 2>/dev/null; then
		cat "${LOG_DIR}/darwin-host.log" >&2 || true
		echo "bpftime-darwin-host exited early" >&2
		exit 1
	fi
	HOST_SHM="$(sed -n 's/.*created SHM segment \(bpftime_darwin_shm_[0-9]*\).*/\1/p' "${LOG_DIR}/darwin-host.log" | head -n 1)"
	if [[ -n "${HOST_SHM}" ]]; then
		break
	fi
	sleep 0.2
done
if [[ -z "${HOST_SHM}" ]]; then
	cat "${LOG_DIR}/darwin-host.log" >&2 || true
	echo "failed to detect SHM segment from bpftime-darwin-host" >&2
	exit 1
fi

echo "mode=macos-cross-end bridge=${GW_IP}:${PORT} host_shm=${HOST_SHM} log_dir=${LOG_DIR}" >&2

# 2. Start unmodified arm64 Linux OTel eBPF collector inside the VM.
VM_SHM="bpftime_otel_vm_$$"
vssh "pkill -9 -f '[o]telcol-ebpf-profiler' 2>/dev/null || true" 2>/dev/null || true
vssh "rm -f /dev/shm/bpftime_otel_vm_* 2>/dev/null || true" 2>/dev/null || true
vssh "env \
	BPFTIME_GO_SECCOMP=1 \
	BPFTIME_DARWIN_HOST='${GW_IP}:${PORT}' \
	BPFTIME_TARGET_OSX=1 \
	BPFTIME_GLOBAL_SHM_NAME='${VM_SHM}' \
	BPFTIME_SHM_MEMORY_MB=512 \
	BPFTIME_MAX_FD_COUNT=65536 \
	BPFTIME_LOG_OUTPUT='${LOG_DIR}/bpftime-vm-collector.log' \
	SPDLOG_LEVEL=info \
	LD_PRELOAD='${VM_SERVER_SO}' \
	OTEL_MALLOC_PROBE='uprobe:/lib/aarch64-linux-gnu/libc.so.6:malloc' \
	OTEL_FREE_PROBE='uprobe:/lib/aarch64-linux-gnu/libc.so.6:free' \
	'${COLLECTOR}' --feature-gates=+service.profilesSupport \
		--set=service.telemetry.metrics.level=none \
		--set=receivers.profiling.no_kernel_version_check=true \
		--config '${CONFIG}'" >"${LOG_DIR}/collector.log" 2>&1 &
VM_COLLECTOR_LAUNCH_PID=$!

for ((i = 0; i < STARTUP_WAIT; i++)); do
	if ! kill -0 "${VM_COLLECTOR_LAUNCH_PID}" 2>/dev/null; then
		tail -n 120 "${LOG_DIR}/collector.log" >&2 || true
		tail -n 80 "${LOG_DIR}/darwin-host.log" >&2 || true
		echo "VM OTel collector exited during startup" >&2
		exit 1
	fi
	if grep -Fq "Attached probes" "${LOG_DIR}/collector.log" 2>/dev/null; then
		break
	fi
	sleep 1
done

if ! grep -Fq "Attached probes" "${LOG_DIR}/collector.log" 2>/dev/null; then
	tail -n 120 "${LOG_DIR}/collector.log" >&2 || true
	tail -n 80 "${LOG_DIR}/darwin-host.log" >&2 || true
	echo "VM OTel collector did not finish attaching probes within ${STARTUP_WAIT}s" >&2
	exit 1
fi

# 3. Run native macOS Mach-O victim with DYLD_INSERT_LIBRARIES=libbpftime-agent.dylib.
env \
	BPFTIME_GLOBAL_SHM_NAME="${HOST_SHM}" \
	BPFTIME_SHM_MEMORY_MB=1024 \
	BPFTIME_LOG_OUTPUT="${LOG_DIR}/bpftime-macos-victim.log" \
	SPDLOG_LEVEL=info \
	DYLD_INSERT_LIBRARIES="${AGENT_DYLIB}" \
	"${VICTIM_MACOS}" >"${LOG_DIR}/victim.log" 2>&1 &
VICTIM_PID=$!
MACOS_VICTIM_PID="${VICTIM_PID}"
echo "native macOS Mach-O victim pid=${MACOS_VICTIM_PID}" >&2

sleep "${RUN_SECONDS}"
kill "${VICTIM_PID}" 2>/dev/null || true
wait "${VICTIM_PID}" 2>/dev/null || true
VICTIM_PID=""
sleep "${FLUSH_WAIT}"

# 4. Assertions.
if [[ "$(grep -Ec 'Created uprobe/uretprobe perf event handler, module name .*libsystem_malloc\.dylib' "${LOG_DIR}/darwin-host.log" || true)" -lt 2 ]]; then
	echo "darwin-host did not create both libsystem_malloc.dylib uprobe handlers" >&2
	tail -n 80 "${LOG_DIR}/darwin-host.log" >&2 || true
	exit 1
fi

if ! grep -Fq "Attach successfully" "${LOG_DIR}"/bpftime-*log* "${LOG_DIR}/darwin-host.log" 2>/dev/null; then
	echo "macOS libbpftime-agent.dylib did not attach inside native victim" >&2
	tail -n 80 "${LOG_DIR}/bpftime-macos-victim.log" >&2 || true
	exit 1
fi

if ! grep -Fq "Profiles" "${LOG_DIR}/collector.log" ||
	! grep -Fq "process.executable.name: Str(victim-macos)" "${LOG_DIR}/collector.log" ||
	! grep -Fq "process.pid: Int(${MACOS_VICTIM_PID})" "${LOG_DIR}/collector.log"; then
	echo "VM OTel collector did not export a profile for native macOS victim (pid=${MACOS_VICTIM_PID})" >&2
	tail -n 120 "${LOG_DIR}/collector.log" >&2 || true
	tail -n 80 "${LOG_DIR}/darwin-host.log" >&2 || true
	exit 1
fi

echo "PASS (macos-cross-end/rootless): OTel collector in ARM64 Linux VM exported profile for native macOS Mach-O victim-macos (pid=${MACOS_VICTIM_PID})"
