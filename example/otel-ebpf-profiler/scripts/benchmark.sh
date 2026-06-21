#!/usr/bin/env bash
# Produce a small local CSV comparing the malloc/free workload with and without
# bpftime. Set RUN_FULL_OTEL=1 to include the upstream OTel collector profiler.

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/.run-logs/benchmark-$(date +%Y%m%d-%H%M%S)}"
RESULTS="${RESULTS:-${LOG_DIR}/results.csv}"
ITERATIONS="${ITERATIONS:-200000}"
REPEATS="${REPEATS:-5}"
TIME_CMD="${TIME_CMD:-/usr/bin/time}"
BPFTIME_AGENT="${BPFTIME_AGENT:-${REPO_ROOT}/build/runtime/agent/libbpftime-agent.so}"

need_file "${TIME_CMD}" "time command"
need_file "${BPFTIME_AGENT}" "bpftime agent"
ensure_sudo
mkdir -p "${LOG_DIR}"
make -C "${EXAMPLE_DIR}" victim tracer >/dev/null

printf 'case,run,elapsed_seconds,user_seconds,sys_seconds,max_rss_kb\n' >"${RESULTS}"

record_time() {
	local name="$1"
	local run="$2"
	local out_file="${LOG_DIR}/${name}-${run}.out"
	shift 2
	log "benchmark ${name} run ${run}/${REPEATS}"
	"${TIME_CMD}" -q -f "${name},${run},%e,%U,%S,%M" -a -o "${RESULTS}" "$@" \
		>"${out_file}" 2>&1
}

run_baseline() {
	local run="$1"
	record_time baseline "${run}" sudo env \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
}

run_agent_only() {
	local run="$1"
	BPFTIME_GLOBAL_SHM_NAME="bpftime_maps_shm_otel_bench_agent_${run}_$$"
	export BPFTIME_GLOBAL_SHM_NAME
	start_bpftime_daemon "${LOG_DIR}/bpftime-agent-${run}.log"
	record_time bpftime-agent "${run}" sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		LD_PRELOAD="${BPFTIME_AGENT}" \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	BPFTIME_DAEMON_PID=
	cleanup_shm
}

run_minimal_loader() {
	local run="$1"
	BPFTIME_GLOBAL_SHM_NAME="bpftime_maps_shm_otel_bench_minimal_${run}_$$"
	export BPFTIME_GLOBAL_SHM_NAME
	start_bpftime_daemon "${LOG_DIR}/bpftime-minimal-${run}.log"
	sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		"${EXAMPLE_DIR}/otel_malloc_free" \
		-object "${EXAMPLE_DIR}/malloc_free.bpf.o" \
		-duration "${TRACER_DURATION:-120s}" \
		-interval "${TRACER_INTERVAL:-30s}" >"${LOG_DIR}/minimal-loader-${run}.log" 2>&1 &
	TRACER_PID=$!
	sleep "${TRACER_STARTUP_DELAY:-2}"
	if ! kill -0 "${TRACER_PID}" 2>/dev/null; then
		tail -n 80 "${LOG_DIR}/minimal-loader-${run}.log" >&2 || true
		die "minimal loader exited early"
	fi
	record_time minimal-cilium-loader "${run}" sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		LD_PRELOAD="${BPFTIME_AGENT}" \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
	stop_pid "${TRACER_PID:-}"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	TRACER_PID=
	BPFTIME_DAEMON_PID=
	cleanup_shm
}

run_full_otel() {
	local run="$1"
	local otel_dir="${OTEL_EBPF_PROFILER_DIR:-${EXAMPLE_DIR}/.otel-ebpf-profiler}"
	local otelcol="${OTEL_COLLECTOR_BIN:-${otel_dir}/otelcol-ebpf-profiler}"
	local config="${OTEL_COLLECTOR_CONFIG:-${EXAMPLE_DIR}/config/otelcol-malloc-free.yaml}"
	need_file "${otelcol}" "otelcol-ebpf-profiler"
	need_file "${config}" "collector config"

	local libc
	libc="$(find_libc)"
	export OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE:-uprobe:${libc}:malloc}"
	export OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE:-uprobe:${libc}:free}"
	BPFTIME_GLOBAL_SHM_NAME="bpftime_maps_shm_otel_bench_full_${run}_$$"
	export BPFTIME_GLOBAL_SHM_NAME

	start_bpftime_daemon "${LOG_DIR}/bpftime-full-otel-${run}.log"
	sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		OTEL_EBPF_PROFILER_MALLOC_PROBE="${OTEL_EBPF_PROFILER_MALLOC_PROBE}" \
		OTEL_EBPF_PROFILER_FREE_PROBE="${OTEL_EBPF_PROFILER_FREE_PROBE}" \
		"${otelcol}" \
		--feature-gates=+service.profilesSupport \
		--config "${config}" >"${LOG_DIR}/otelcol-${run}.log" 2>&1 &
	OTELCOL_PID=$!
	sleep "${OTEL_COLLECTOR_STARTUP_DELAY:-5}"
	if ! kill -0 "${OTELCOL_PID}" 2>/dev/null; then
		tail -n 120 "${LOG_DIR}/otelcol-${run}.log" >&2 || true
		die "otelcol-ebpf-profiler exited early"
	fi
	record_time full-otel-collector "${run}" sudo env \
		BPFTIME_GLOBAL_SHM_NAME="${BPFTIME_GLOBAL_SHM_NAME}" \
		LD_PRELOAD="${BPFTIME_AGENT}" \
		VICTIM_ITERATIONS="${ITERATIONS}" \
		VICTIM_SLEEP_US=0 \
		VICTIM_PRINT_EVERY=0 \
		"${EXAMPLE_DIR}/victim"
	sleep "${OTEL_FLUSH_DELAY:-6}"
	stop_pid "${OTELCOL_PID:-}"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	OTELCOL_PID=
	BPFTIME_DAEMON_PID=
	cleanup_shm
}

cleanup() {
	stop_pid "${TRACER_PID:-}"
	stop_pid "${OTELCOL_PID:-}"
	stop_pid "${BPFTIME_DAEMON_PID:-}"
	cleanup_shm
}
trap cleanup EXIT INT TERM

for run in $(seq 1 "${REPEATS}"); do
	run_baseline "${run}"
	run_agent_only "${run}"
	run_minimal_loader "${run}"
	if [[ "${RUN_FULL_OTEL:-0}" == "1" ]]; then
		run_full_otel "${run}"
	fi
done

log "results: ${RESULTS}"
awk -F, 'NR > 1 {count[$1]++; sum[$1] += $3}
	END {
		print "case,avg_elapsed_seconds";
		for (name in count) {
			printf "%s,%.6f\n", name, sum[name] / count[name];
		}
	}' "${RESULTS}" | {
	read -r header
	printf '%s\n' "${header}"
	sort
} >&2
