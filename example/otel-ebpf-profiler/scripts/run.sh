#!/usr/bin/env bash
# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#
# Runs the unmodified upstream OpenTelemetry eBPF profiler against the malloc/free
# uprobes of a looping victim.
# MODE=userspace (default): rootless pure userspace via LD_PRELOAD of the
#   syscall-server (collector) and agent (victim); no root, no kernel eBPF.
# MODE=daemon: legacy kernel-mirror path requiring root and bpftime_daemon.

set -euo pipefail

MODE="${MODE:-userspace}"
RUN_SECONDS="${RUN_SECONDS:-12}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${BPFTIME_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}/build}"
SERVER_SO="${BUILD_DIR}/runtime/syscall-server/libbpftime-syscall-server.so"
AGENT_SO="${BUILD_DIR}/runtime/agent/libbpftime-agent.so"
DAEMON="${BPFTIME_DAEMON:-${BUILD_DIR}/daemon/bpftime_daemon}"
COLLECTOR="${OTEL_COLLECTOR_BIN:-}"
CONFIG="${EXAMPLE_DIR}/config/otelcol.yaml"
VICTIM="${EXAMPLE_DIR}/victim"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/$(date +%Y%m%d-%H%M%S)}"
SHM_NAME="bpftime_otel_profiler_$$"

[[ -x "${COLLECTOR}" ]] || { echo "set OTEL_COLLECTOR_BIN to the built otelcol-ebpf-profiler executable" >&2; exit 1; }
[[ -x "${VICTIM}" ]] || cc -O2 -g -Wall -Wextra "${EXAMPLE_DIR}/victim-src/victim.c" -o "${VICTIM}"

# Locate the victim glibc: LIBC_PATH override, then ldd, then per-arch paths.
detect_libc() {
	[[ -n "${LIBC_PATH:-}" && -e "${LIBC_PATH}" ]] && { echo "${LIBC_PATH}"; return; }
	local p
	p="$(ldd "${VICTIM}" 2>/dev/null | awk '/libc\.so\.6 =>/{print $3; exit}')"
	[[ -n "${p}" && -e "${p}" ]] && { echo "${p}"; return; }
	case "$(uname -m)" in
		x86_64 | amd64) set -- /lib/x86_64-linux-gnu/libc.so.6 /usr/lib64/libc.so.6 ;;
		aarch64 | arm64) set -- /lib/aarch64-linux-gnu/libc.so.6 /usr/lib/aarch64-linux-gnu/libc.so.6 /usr/lib64/libc.so.6 ;;
		*) set -- /usr/lib64/libc.so.6 /usr/lib/libc.so.6 ;;
	esac
	local c; for c in "$@"; do [[ -e "${c}" ]] && { echo "${c}"; return; }; done
	return 1
}
LIBC="$(detect_libc)" || { echo "could not locate libc.so.6; set LIBC_PATH=/path/to/libc.so.6" >&2; exit 1; }

mkdir -p "${LOG_DIR}"; COLLECTOR_PID="" DAEMON_PID="" VICTIM_PID=""
cleanup() {
	local status=$?
	trap - EXIT INT TERM
	# shellcheck disable=SC2086
	kill ${VICTIM_PID} ${COLLECTOR_PID} ${DAEMON_PID} 2>/dev/null || true
	# shellcheck disable=SC2086
	wait ${VICTIM_PID} ${COLLECTOR_PID} ${DAEMON_PID} 2>/dev/null || true
	rm -f "/dev/shm/${SHM_NAME}" "/dev/shm/sem.${SHM_NAME}"
	echo "logs: ${LOG_DIR}" >&2
	exit "${status}"
}
trap cleanup EXIT INT TERM

# Both processes resolve the co-located bpftime libraries and share one SHM.
LD_PATH="${BUILD_DIR}/runtime/syscall-server:${BUILD_DIR}/runtime/agent:${BUILD_DIR}/attach/text_segment_transformer${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_userspace() {
	for f in "${SERVER_SO}" "${AGENT_SO}" "${CONFIG}"; do [[ -e "${f}" ]] || { echo "required build artifact not found: ${f}" >&2; exit 1; }; done
	# BPFTIME_GO_SECCOMP arms seccomp USER_NOTIF from a loader constructor to
	# catch the collector's raw bpf(2); --set skips the kernel version gate.
	BPFTIME_GO_SECCOMP=1 BPFTIME_GLOBAL_SHM_NAME="${SHM_NAME}" \
	BPFTIME_SHM_MEMORY_MB="${BPFTIME_SHM_MEMORY_MB:-512}" BPFTIME_MAX_FD_COUNT=65536 \
	BPFTIME_LOG_OUTPUT="${LOG_DIR}/bpftime-collector.log" SPDLOG_LEVEL=info \
	LD_LIBRARY_PATH="${LD_PATH}" LD_PRELOAD="${SERVER_SO}" \
	OTEL_MALLOC_PROBE="uprobe:${LIBC}:malloc" OTEL_FREE_PROBE="uprobe:${LIBC}:free" \
		"${COLLECTOR}" --feature-gates=+service.profilesSupport \
			--set=receivers.profiling.no_kernel_version_check=true \
			--config "${CONFIG}" >"${LOG_DIR}/collector.log" 2>&1 &
	COLLECTOR_PID=$!
	# Wait until both uprobe handlers and both BPF links land in shared memory.
	local i
	for ((i = 0; i < 20; i++)); do
		kill -0 "${COLLECTOR_PID}" 2>/dev/null || { tail -n 80 "${LOG_DIR}/collector.log" >&2 || true; echo "OpenTelemetry collector exited during startup" >&2; exit 1; }
		[[ "$(grep -Ec 'Created uprobe/uretprobe perf event handler, module name .*libc\.so\.6,' "${LOG_DIR}/bpftime-collector.log" 2>/dev/null || true)" -ge 2 ]] &&
			grep -q "Attached probes" "${LOG_DIR}/collector.log" 2>/dev/null && break
		sleep 1
	done
	# Data plane: the agent enumerates published handlers and attaches via Frida.
	timeout "${RUN_SECONDS}" env BPFTIME_GLOBAL_SHM_NAME="${SHM_NAME}" \
		BPFTIME_SHM_MEMORY_MB="${BPFTIME_SHM_MEMORY_MB:-512}" \
		BPFTIME_LOG_OUTPUT="${LOG_DIR}/bpftime-victim.log" SPDLOG_LEVEL=info \
		LD_LIBRARY_PATH="${LD_PATH}" LD_PRELOAD="${AGENT_SO}" \
		"${VICTIM}" >"${LOG_DIR}/victim.log" 2>&1 &
	VICTIM_PID=$!
	wait "${VICTIM_PID}" || true
	VICTIM_PID=""
	sleep 6
	kill -0 "${COLLECTOR_PID}" 2>/dev/null || { echo "collector exited before exporting a profile" >&2; exit 1; }
	[[ "$(grep -Eh 'Created uprobe/uretprobe perf event handler, module name .*libc\.so\.6,' "${LOG_DIR}"/bpftime-collector*log* 2>/dev/null | wc -l)" -ge 2 ]] ||
		{ echo "bpftime did not create both malloc/free uprobe handlers" >&2; exit 1; }
	grep -Fq "Attached probes" "${LOG_DIR}/collector.log" ||
		{ echo "bpftime did not emulate both profiler BPF_LINK_CREATE links" >&2; exit 1; }
	grep -Fq "Attach successfully" "${LOG_DIR}"/bpftime-*log* 2>/dev/null ||
		{ echo "bpftime agent did not attach inside the victim" >&2; exit 1; }
	grep -q "Profiles" "${LOG_DIR}/collector.log" && grep -q "process.executable.name: Str(victim)" "${LOG_DIR}/collector.log" ||
		{ echo "collector did not export a profile for the victim" >&2; exit 1; }
	echo "PASS (userspace/rootless): bpftime ran the profiler in userspace and exported a victim profile"
}

run_daemon() {
	[[ "${EUID}" -eq 0 ]] || { echo "MODE=daemon mirrors kernel links and must be run as root" >&2; exit 1; }
	[[ -x "${DAEMON}" ]] || { echo "bpftime_daemon not found: ${DAEMON}" >&2; exit 1; }
	BPFTIME_GLOBAL_SHM_NAME="${SHM_NAME}" BPFTIME_MAX_FD_COUNT=65536 \
		"${DAEMON}" -v >"${LOG_DIR}/daemon.log" 2>&1 &
	DAEMON_PID=$!
	sleep 2
	kill -0 "${DAEMON_PID}" 2>/dev/null || { echo "bpftime daemon exited during startup" >&2; exit 1; }
	OTEL_MALLOC_PROBE="uprobe:${LIBC}:malloc" OTEL_FREE_PROBE="uprobe:${LIBC}:free" \
		"${COLLECTOR}" --feature-gates=+service.profilesSupport \
			--config "${CONFIG}" >"${LOG_DIR}/collector.log" 2>&1 &
	COLLECTOR_PID=$!
	sleep 5
	kill -0 "${COLLECTOR_PID}" 2>/dev/null || { tail -n 80 "${LOG_DIR}/collector.log" >&2; echo "OpenTelemetry collector exited during startup" >&2; exit 1; }
	timeout "${RUN_SECONDS}" "${VICTIM}" >"${LOG_DIR}/victim.log" 2>&1 || true
	sleep 5
	kill -0 "${DAEMON_PID}" 2>/dev/null && kill -0 "${COLLECTOR_PID}" 2>/dev/null ||
		{ echo "daemon or collector exited during the run" >&2; exit 1; }
	local pat
	for pat in 'cmd:BPF_LINK_CREATE' "Created uprobe/uretprobe perf event handler, module name ${LIBC}," 'attach perf [0-9].* to bpf [0-9].*, for pid'; do
		[[ "$(grep -Ec "${pat}" "${LOG_DIR}/daemon.log")" -ge 2 ]] ||
			{ echo "bpftime did not mirror both profiler links and uprobes: ${pat}" >&2; exit 1; }
	done
	grep -q 'Profiles' "${LOG_DIR}/collector.log" && grep -q 'process.executable.name: Str(victim)' "${LOG_DIR}/collector.log" ||
		{ echo "collector did not export a profile for the victim" >&2; exit 1; }
	echo "PASS (daemon): mirrored malloc/free profiler links and observed victim profile"
}

echo "mode=${MODE} libc=${LIBC} shm=${SHM_NAME} log_dir=${LOG_DIR}" >&2
case "${MODE}" in
	userspace) run_userspace ;;
	daemon) run_daemon ;;
	*) echo "unknown MODE='${MODE}' (expected userspace or daemon)" >&2; exit 1 ;;
esac
