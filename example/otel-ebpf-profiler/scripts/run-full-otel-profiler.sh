#!/usr/bin/env bash
# Run the real OpenTelemetry eBPF profiler collector receiver with malloc/free
# probe_links, using bpftime for the target process.

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/full-otel-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

OTEL_EBPF_PROFILER_DIR="${OTEL_EBPF_PROFILER_DIR:-${EXAMPLE_DIR}/.otel-ebpf-profiler}"
OTEL_COLLECTOR_BIN="${OTEL_COLLECTOR_BIN:-${OTEL_EBPF_PROFILER_DIR}/otelcol-ebpf-profiler}"
OTEL_COLLECTOR_CONFIG="${OTEL_COLLECTOR_CONFIG:-${EXAMPLE_DIR}/config/otelcol-malloc-free.yaml}"
BPFTIME_AGENT="${BPFTIME_AGENT:-${REPO_ROOT}/build/runtime/agent/libbpftime-agent.so}"
RUN_SECONDS="${RUN_SECONDS:-20}"

need_file "${OTEL_COLLECTOR_BIN}" "otelcol-ebpf-profiler"
need_file "${OTEL_COLLECTOR_CONFIG}" "collector config"
need_file "${BPFTIME_AGENT}" "bpftime agent"

ensure_sudo
make -C "${EXAMPLE_DIR}" victim >/dev/null

LIBC_PATH="$(find_libc)"
export OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE:-uprobe:${LIBC_PATH}:malloc}"
export OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE:-uprobe:${LIBC_PATH}:free}"
: "${BPFTIME_GLOBAL_SHM_NAME:=bpftime_maps_shm_otel_full_$$}"
export BPFTIME_GLOBAL_SHM_NAME

cleanup() {
	local status=$?
	stop_pid "${VICTIM_PID:-}"
	stop_pid "${OTELCOL_PID:-}"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	cleanup_shm
	log "logs: ${LOG_DIR}"
	exit "${status}"
}
trap cleanup EXIT INT TERM

start_bpftime_daemon "${LOG_DIR}/bpftime-daemon.log"

log "starting upstream OTel collector profiler"
sudo env \
	BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
	OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE}" \
	OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE}" \
	"${OTEL_COLLECTOR_BIN}" \
	--feature-gates=+service.profilesSupport \
	--config "${OTEL_COLLECTOR_CONFIG}" >"${LOG_DIR}/otelcol.log" 2>&1 &
OTELCOL_PID=$!

sleep "${OTEL_COLLECTOR_STARTUP_DELAY:-5}"
if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
	tail -n 120 "${LOG_DIR}/otelcol.log" >&2 || true
	die "otelcol-ebpf-profiler exited before the workload started"
fi

log "running victim for ${RUN_SECONDS}s with bpftime agent"
set +e
sudo env \
	BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
	LD_PRELOAD="${BPFTIME_AGENT}" \
	VICTIM_SLEEP_US="${VICTIM_SLEEP_US:-0}" \
	VICTIM_PRINT_EVERY="${VICTIM_PRINT_EVERY:-0}" \
	timeout "${RUN_SECONDS}" "${EXAMPLE_DIR}/victim" >"${LOG_DIR}/victim.log" 2>&1
victim_status=$?
set -e
if [[ "${victim_status}" -ne 0 && "${victim_status}" -ne 124 ]]; then
	tail -n 80 "${LOG_DIR}/victim.log" >&2 || true
	die "victim failed with status ${victim_status}"
fi

log "waiting for profiler flush"
sleep "${OTEL_FLUSH_DELAY:-6}"

if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
	tail -n 120 "${LOG_DIR}/otelcol.log" >&2 || true
	die "otelcol-ebpf-profiler exited during the run"
fi

log "probe links:"
log "  ${OTEL_EBPF_PROFILER_MALLOC_PROBE}"
log "  ${OTEL_EBPF_PROFILER_FREE_PROBE}"
log "recent collector output:"
grep -Ei 'profile|sample|probe|error|failed|starting' "${LOG_DIR}/otelcol.log" | tail -n 40 >&2 || true
