#!/usr/bin/env bash
# Fetch and build the upstream OpenTelemetry eBPF profiler binaries used by this
# example. The checkout is intentionally external to bpftime source files.

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OTEL_EBPF_PROFILER_REPO="${OTEL_EBPF_PROFILER_REPO:-https://github.com/open-telemetry/opentelemetry-ebpf-profiler.git}"
OTEL_EBPF_PROFILER_REF="${OTEL_EBPF_PROFILER_REF:-main}"
OTEL_EBPF_PROFILER_DIR="${OTEL_EBPF_PROFILER_DIR:-${EXAMPLE_DIR}/.otel-ebpf-profiler}"
OTEL_EBPF_PROFILER_BUILD_TARGETS="${OTEL_EBPF_PROFILER_BUILD_TARGETS:-otelcol-ebpf-profiler}"

if [[ ! -d "${OTEL_EBPF_PROFILER_DIR}/.git" ]]; then
	log "cloning ${OTEL_EBPF_PROFILER_REPO} into ${OTEL_EBPF_PROFILER_DIR}"
	git clone "${OTEL_EBPF_PROFILER_REPO}" "${OTEL_EBPF_PROFILER_DIR}"
else
	log "updating ${OTEL_EBPF_PROFILER_DIR}"
	git -C "${OTEL_EBPF_PROFILER_DIR}" fetch --tags origin
fi

if git -C "${OTEL_EBPF_PROFILER_DIR}" rev-parse --verify --quiet "origin/${OTEL_EBPF_PROFILER_REF}" >/dev/null; then
	git -C "${OTEL_EBPF_PROFILER_DIR}" checkout -B "${OTEL_EBPF_PROFILER_REF}" "origin/${OTEL_EBPF_PROFILER_REF}"
else
	git -C "${OTEL_EBPF_PROFILER_DIR}" checkout "${OTEL_EBPF_PROFILER_REF}"
fi

export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
read -r -a build_targets <<<"${OTEL_EBPF_PROFILER_BUILD_TARGETS}"

log "building upstream target(s): ${build_targets[*]}"
make -C "${OTEL_EBPF_PROFILER_DIR}" "${build_targets[@]}"

log "upstream checkout: ${OTEL_EBPF_PROFILER_DIR}"
for target in "${build_targets[@]}"; do
	if [[ -x "${OTEL_EBPF_PROFILER_DIR}/${target}" ]]; then
		log "built ${OTEL_EBPF_PROFILER_DIR}/${target}"
	fi
done
