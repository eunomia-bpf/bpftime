#!/usr/bin/env bash
# Shared helpers for the OpenTelemetry eBPF profiler bpftime example.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${EXAMPLE_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${EXAMPLE_DIR}/../.." && pwd))"

log() {
	printf '[otel-bpftime] %s\n' "$*" >&2
}

die() {
	printf '[otel-bpftime] error: %s\n' "$*" >&2
	exit 1
}

need_file() {
	local path="$1"
	local name="$2"
	[[ -e "${path}" ]] || die "${name} not found: ${path}"
}

find_libc() {
	if [[ -n "${LIBC_PATH:-}" ]]; then
		need_file "${LIBC_PATH}" "libc"
		printf '%s\n' "${LIBC_PATH}"
		return
	fi

	local candidates=(
		/lib/x86_64-linux-gnu/libc.so.6
		/usr/lib/x86_64-linux-gnu/libc.so.6
		/lib64/libc.so.6
		/usr/lib64/libc.so.6
		/lib/aarch64-linux-gnu/libc.so.6
		/usr/lib/aarch64-linux-gnu/libc.so.6
		/lib/riscv64-linux-gnu/libc.so.6
		/usr/lib/riscv64-linux-gnu/libc.so.6
	)

	local path
	for path in "${candidates[@]}"; do
		if [[ -e "${path}" ]]; then
			printf '%s\n' "${path}"
			return
		fi
	done

	if command -v ldconfig >/dev/null 2>&1; then
		path="$(ldconfig -p | awk '/libc\.so\.6/ {print $NF; exit}')"
		if [[ -n "${path}" && -e "${path}" ]]; then
			printf '%s\n' "${path}"
			return
		fi
	fi

	die "libc.so.6 not found; set LIBC_PATH=/path/to/libc.so.6"
}

ensure_sudo() {
	if [[ "${BPFTIME_SKIP_SUDO_VALIDATE:-0}" != "1" ]]; then
		if sudo -n true 2>/dev/null; then
			return
		fi
		if [[ -t 0 ]]; then
			sudo -v
			return
		fi
		die "sudo credentials are required; run sudo -v in a terminal first"
	fi
}

stop_pid() {
	local pid="${1:-}"
	[[ -n "${pid}" ]] || return 0
	if kill -0 "${pid}" 2>/dev/null; then
		sudo kill "${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
		wait "${pid}" 2>/dev/null || true
	fi
}

cleanup_shm() {
	if [[ -n "${BPFTIME_GLOBAL_SHM_NAME:-}" && "${BPFTIME_KEEP_SHM:-0}" != "1" ]]; then
		sudo rm -f "/dev/shm/${BPFTIME_GLOBAL_SHM_NAME}" \
			"/dev/shm/sem.${BPFTIME_GLOBAL_SHM_NAME}" 2>/dev/null || true
	fi
}

start_bpftime_daemon() {
	local log_file="$1"
	local daemon="${BPFTIME_DAEMON:-${REPO_ROOT}/build/daemon/bpftime_daemon}"

	need_file "${daemon}" "bpftime daemon"
	: "${BPFTIME_GLOBAL_SHM_NAME:=bpftime_maps_shm_otel_$$}"
	export BPFTIME_GLOBAL_SHM_NAME

	log "starting bpftime daemon (shm=${BPFTIME_GLOBAL_SHM_NAME})"
	sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		BPFTIME_SHM_MEMORY_MB="${BPFTIME_SHM_MEMORY_MB:-256}" \
		SPDLOG_LEVEL="${SPDLOG_LEVEL:-info}" \
		"${daemon}" -v >"${log_file}" 2>&1 &
	BPFTIME_DAEMON_PID=$!
	export BPFTIME_DAEMON_PID

	sleep "${BPFTIME_DAEMON_STARTUP_DELAY:-1}"
	if ! kill -0 "${BPFTIME_DAEMON_PID}" 2>/dev/null; then
		tail -n 100 "${log_file}" >&2 || true
		die "bpftime daemon exited early"
	fi
}
