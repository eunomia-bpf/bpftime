#!/usr/bin/env bash
# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
	echo "run this script as root" >&2
	exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${EXAMPLE_DIR}" rev-parse --show-toplevel)"
DAEMON="${BPFTIME_DAEMON:-${REPO_ROOT}/build/daemon/bpftime_daemon}"
COLLECTOR="${OTEL_COLLECTOR_BIN:-}"
CONFIG="${EXAMPLE_DIR}/config/otelcol.yaml"
VICTIM="${EXAMPLE_DIR}/victim"
RUN_SECONDS="${RUN_SECONDS:-12}"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/$(date +%Y%m%d-%H%M%S)}"
SHM_NAME="bpftime_otel_profiler_$$"

for file in "${DAEMON}" "${COLLECTOR}" "${VICTIM}"; do
	if [[ ! -x "${file}" ]]; then
		echo "required executable not found: ${file:-OTEL_COLLECTOR_BIN}" >&2
		exit 1
	fi
done

LIBC="${LIBC_PATH:-/lib/x86_64-linux-gnu/libc.so.6}"
if [[ ! -e "${LIBC}" ]]; then
	echo "libc not found; set LIBC_PATH=/path/to/libc.so.6" >&2
	exit 1
fi

mkdir -p "${LOG_DIR}"

cleanup() {
	local status=$?
	trap - EXIT INT TERM
	kill "${COLLECTOR_PID:-}" "${DAEMON_PID:-}" 2>/dev/null || true
	wait "${COLLECTOR_PID:-}" "${DAEMON_PID:-}" 2>/dev/null || true
	rm -f "/dev/shm/${SHM_NAME}" "/dev/shm/sem.${SHM_NAME}"
	echo "logs: ${LOG_DIR}" >&2
	exit "${status}"
}
trap cleanup EXIT INT TERM

BPFTIME_GLOBAL_SHM_NAME="${SHM_NAME}" \
BPFTIME_MAX_FD_COUNT=65536 \
"${DAEMON}" -v >"${LOG_DIR}/daemon.log" 2>&1 &
DAEMON_PID=$!

sleep 2
if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
	echo "bpftime daemon exited during startup" >&2
	exit 1
fi

OTEL_MALLOC_PROBE="uprobe:${LIBC}:malloc" \
OTEL_FREE_PROBE="uprobe:${LIBC}:free" \
"${COLLECTOR}" --feature-gates=+service.profilesSupport \
	--config "${CONFIG}" >"${LOG_DIR}/collector.log" 2>&1 &
COLLECTOR_PID=$!

sleep 5
if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
	tail -n 80 "${LOG_DIR}/collector.log" >&2
	echo "OpenTelemetry collector exited during startup" >&2
	exit 1
fi

timeout "${RUN_SECONDS}" "${VICTIM}" >"${LOG_DIR}/victim.log" 2>&1 ||
	[[ "$?" -eq 124 ]]
sleep 5

if ! kill -0 "${DAEMON_PID}" 2>/dev/null ||
	! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
	echo "daemon or collector exited during the run" >&2
	exit 1
fi
if [[ "$(grep -c 'cmd:BPF_LINK_CREATE' "${LOG_DIR}/daemon.log")" -lt 2 ]]; then
	echo "bpftime did not mirror both profiler links" >&2
	exit 1
fi
if [[ "$(grep -c 'Created uprobe/uretprobe perf event handler' "${LOG_DIR}/daemon.log")" -lt 2 ]]; then
	echo "bpftime did not create both profiler uprobe handlers" >&2
	exit 1
fi
if [[ "$(grep -c 'attach perf [0-9].* to bpf' "${LOG_DIR}/daemon.log")" -lt 2 ]]; then
	echo "bpftime did not attach both profiler uprobes" >&2
	exit 1
fi
if ! grep -q 'Profiles' "${LOG_DIR}/collector.log" ||
	! grep -q 'process.executable.name: Str(victim)' "${LOG_DIR}/collector.log"; then
	echo "collector did not export a profile for the victim" >&2
	exit 1
fi

echo "PASS: mirrored malloc/free profiler links and observed victim profile"
