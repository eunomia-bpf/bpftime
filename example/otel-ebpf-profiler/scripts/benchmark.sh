#!/usr/bin/env bash
# Benchmark the complete upstream OpenTelemetry eBPF profiler stack collection
# path on the malloc/free workload. This script intentionally does not run any
# local BPF loader.

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/otel-stack-benchmark-$(date +%Y%m%d-%H%M%S)}"
RESULTS="${RESULTS:-${LOG_DIR}/results.csv}"
ITERATIONS="${ITERATIONS:-0}"
RUN_SECONDS="${RUN_SECONDS:-10}"
REPEATS="${REPEATS:-3}"
TIME_CMD="${TIME_CMD:-/usr/bin/time}"
OTEL_EBPF_PROFILER_DIR="${OTEL_EBPF_PROFILER_DIR:-${EXAMPLE_DIR}/.otel-ebpf-profiler}"
OTEL_COLLECTOR_BIN="${OTEL_COLLECTOR_BIN:-${OTEL_EBPF_PROFILER_DIR}/otelcol-ebpf-profiler}"
OTEL_COLLECTOR_CONFIG="${OTEL_COLLECTOR_CONFIG:-${EXAMPLE_DIR}/config/otelcol-malloc-free.yaml}"
BPFTIME_AGENT="${BPFTIME_AGENT:-${REPO_ROOT}/build/runtime/agent/libbpftime-agent.so}"

need_file "${TIME_CMD}" "time command"
need_file "${OTEL_COLLECTOR_BIN}" "otelcol-ebpf-profiler"
need_file "${OTEL_COLLECTOR_CONFIG}" "collector config"
need_file "${BPFTIME_AGENT}" "bpftime agent"
ensure_sudo
mkdir -p "${LOG_DIR}"
make -C "${EXAMPLE_DIR}" victim >/dev/null

printf 'case,run,elapsed_seconds,user_seconds,sys_seconds,max_rss_kb,iterations,iterations_per_second\n' >"${RESULTS}"

set_otel_probe_env() {
	local libc
	libc="$(find_libc)"
	export OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE:-uprobe:${libc}:malloc}"
	export OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE:-uprobe:${libc}:free}"
}

record_time() {
	local name="$1"
	local run="$2"
	local out_file="${LOG_DIR}/${name}-${run}.out"
	local time_file="${LOG_DIR}/${name}-${run}.time"
	local elapsed user sys rss iterations throughput
	shift 2
	log "benchmark ${name} run ${run}/${REPEATS}"
	"${TIME_CMD}" -q -f "%e,%U,%S,%M" -o "${time_file}" "$@" \
		>"${out_file}" 2>&1
	IFS=, read -r elapsed user sys rss <"${time_file}"
	iterations="$(awk -F= '/^iterations_completed=/ {value=$2} END {print value}' "${out_file}")"
	if [[ -z "${iterations}" ]]; then
		tail -n 80 "${out_file}" >&2 || true
		die "victim did not report iterations_completed"
	fi
	throughput="$(awk -v iterations="${iterations}" -v elapsed="${elapsed}" \
		'BEGIN { if (elapsed > 0) printf "%.6f", iterations / elapsed; else print "0.000000" }')"
	printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
		"${name}" "${run}" "${elapsed}" "${user}" "${sys}" "${rss}" \
		"${iterations}" "${throughput}" >>"${RESULTS}"
}

start_otelcol() {
	local log_file="$1"
	set_otel_probe_env
	log "starting upstream OTel collector profiler"
	sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME:-}" \
		OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE}" \
		OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE}" \
		"${OTEL_COLLECTOR_BIN}" \
		--feature-gates=+service.profilesSupport \
		--config "${OTEL_COLLECTOR_CONFIG}" >"${log_file}" 2>&1 &
	OTELCOL_PID=$!
	export OTELCOL_PID

	sleep "${OTEL_COLLECTOR_STARTUP_DELAY:-5}"
	if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
		tail -n 120 "${log_file}" >&2 || true
		die "otelcol-ebpf-profiler exited before the workload started"
	fi
}

require_profile_output() {
	local log_file="$1"
	if ! grep -q 'Profiles' "${log_file}"; then
		tail -n 160 "${log_file}" >&2 || true
		die "otelcol-ebpf-profiler did not export profiles"
	fi
	if ! grep -Eq 'process\.executable\.name: Str\(victim\)|process\.executable\.path: Str\(.*victim\)' "${log_file}"; then
		tail -n 160 "${log_file}" >&2 || true
		die "otelcol-ebpf-profiler did not export victim profiles"
	fi
}

stop_otelcol() {
	stop_pid "${OTELCOL_PID:-}"
	OTELCOL_PID=
}

run_baseline() {
	local run="$1"
	record_time baseline "${run}" sudo env \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_RUN_SECONDS="${RUN_SECONDS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
}

run_kernel_otel() {
	local run="$1"
	local otel_log="${LOG_DIR}/otel-kernel-${run}.log"
	start_otelcol "${otel_log}"
	record_time otel-kernel-stack-collector "${run}" sudo env \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_RUN_SECONDS="${RUN_SECONDS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
	if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
		tail -n 160 "${otel_log}" >&2 || true
		die "otelcol-ebpf-profiler exited during kernel benchmark"
	fi
	sleep "${OTEL_FLUSH_DELAY:-6}"
	require_profile_output "${otel_log}"
	stop_otelcol
}

run_daemon_mirror_otel() {
	local run="$1"
	local otel_log="${LOG_DIR}/otel-daemon-mirror-${run}.log"
	BPFTIME_GLOBAL_SHM_NAME="bpftime_maps_shm_otel_stack_bench_${run}_$$"
	export BPFTIME_GLOBAL_SHM_NAME
	start_bpftime_daemon "${LOG_DIR}/bpftime-daemon-${run}.log"
	start_otelcol "${otel_log}"
	record_time otel-daemon-mirror-stack-collector "${run}" sudo env \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_RUN_SECONDS="${RUN_SECONDS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		LD_PRELOAD="${BPFTIME_AGENT}" \
		"${EXAMPLE_DIR}/victim"
	if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
		tail -n 160 "${otel_log}" >&2 || true
		die "otelcol-ebpf-profiler exited during daemon mirror benchmark"
	fi
	sleep "${OTEL_FLUSH_DELAY:-6}"
	require_profile_output "${otel_log}"
	stop_otelcol
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	BPFTIME_DAEMON_PID=
	cleanup_shm
}

cleanup() {
	stop_pid "${OTELCOL_PID:-}"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	cleanup_shm
}
trap cleanup EXIT INT TERM

for run in $(seq 1 "${REPEATS}"); do
	run_baseline "${run}"
	run_kernel_otel "${run}"
	run_daemon_mirror_otel "${run}"
done

log "results: ${RESULTS}"
awk -F, 'NR > 1 {count[$1]++; elapsed[$1] += $3; throughput[$1] += $8}
	END {
		print "case,avg_elapsed_seconds,avg_iterations_per_second";
		for (name in elapsed) {
			printf "%s,%.6f,%.6f\n", name, elapsed[name] / count[name], throughput[name] / count[name];
		}
	}' "${RESULTS}" | {
	read -r header
	printf '%s\n' "${header}"
	sort
} >&2
