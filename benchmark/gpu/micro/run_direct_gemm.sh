#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/bpftime_sy03/bpftime"
CLI="$ROOT/build/tools/cli/bpftime"
RUNON="$ROOT/build/tools/bpftimetool/bpftimetool"
PROG="$ROOT/example/gpu/directly_run_on_gpu/directly_run"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ulimit -l unlimited || true
BPFTIME_LOG_OUTPUT=console BPFTIME_SHM_MEMORY_MB=${BPFTIME_SHM_MEMORY_MB:-16} prlimit --memlock=unlimited -- "$CLI" load "$PROG" &
SERVER_PID=$!
sleep 2

# 32x32 -> grid=2x2x1, block=16x16x1
prlimit --memlock=unlimited -- "$RUNON" run-on-cuda cuda__gemm 1 2 2 1 16 16 1
sleep 1
#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/bpftime_sy03/bpftime"
CLI="$ROOT/build/tools/cli/bpftime"
RUNON="$ROOT/build/tools/bpftimetool/bpftimetool"
PROG="$ROOT/example/gpu/directly_run_on_gpu/directly_run"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

BPFTIME_LOG_OUTPUT=console "$CLI" load "$PROG" &
SERVER_PID=$!
sleep 2

# 64x64, grid=4x4x1, block=16x16x1
"$RUNON" run-on-cuda cuda__gemm 1 4 4 1 16 16 1
sleep 1



