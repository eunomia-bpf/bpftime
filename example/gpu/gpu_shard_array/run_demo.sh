#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[demo] build example if needed"
if [ ! -x ./gpu_shard_array ] || [ ! -x ./vec_add ]; then
  make
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[demo][warn] nvidia-smi not found; CUDA may be unavailable" >&2
fi

ROOT=$(cd ../../../ && pwd)
SERVER_SO="$ROOT/build/runtime/syscall-server/libbpftime-syscall-server.so"
AGENT_SO="$ROOT/build/runtime/agent/libbpftime-agent.so"

if [ ! -f "$SERVER_SO" ] || [ ! -f "$AGENT_SO" ]; then
  echo "[demo][error] bpftime runtime not built. Please run: cmake -B build -S . -DBPFTIME_ENABLE_CUDA_ATTACH=ON && cmake --build build -j" >&2
  exit 2
fi

ulimit -l unlimited 2>/dev/null || true

echo "[demo] start gpu_shard_array (syscall-server) in background"
OUT=demo_out.txt
rm -f "$OUT" vec_add.out
HB=1
echo "[demo] HOST_WRITEBACK=$HB"
env SPDLOG_LEVEL=debug BPFTIME_LOG_OUTPUT=console LD_PRELOAD="$SERVER_SO" HOST_WRITEBACK="$HB" stdbuf -oL -eL ./gpu_shard_array > "$OUT" 2>&1 &
SRV_PID=$!

sleep 1
echo "[demo] start vec_add (agent) in background"
env SPDLOG_LEVEL=debug BPFTIME_LOG_OUTPUT=console LD_PRELOAD="$AGENT_SO" stdbuf -oL -eL ./vec_add > vec_add.out 2>&1 &
VEC_PID=$!

trap 'kill ${VEC_PID} ${SRV_PID} >/dev/null 2>&1 || true' EXIT

echo "[demo] checking increments"

# 验证逻辑（轮询文件末尾，直到 DEADLINE 超时）：
# - 首次读到有效值作为 INIT
# - 之后需要 Last >= INIT + REQUIRED_DELTA 才 PASS
# - 超时则 FAIL

DEADLINE=${DEADLINE:-25}
REQUIRED_DELTA=${REQUIRED_DELTA:-3}
START_TS=$SECONDS
INIT=""
LAST=0

is_num() {
  case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac
}

pass=0
while (( SECONDS - START_TS < DEADLINE )); do
  VAL=$(awk -F= '/counter\[0\]=/ {v=$2} END {print v}' "$OUT" 2>/dev/null)
  if is_num "$VAL"; then
    if [ -z "$INIT" ]; then
      INIT=$VAL
    fi
    LAST=$VAL
    if [ "$LAST" -ge $(( INIT + REQUIRED_DELTA )) ]; then
      pass=1
      break
    fi
  fi
  sleep 1
 done

echo "First=${INIT:-0} Last=${LAST:-0}"
if [ "$pass" -eq 1 ]; then
  echo "[demo] PASS: counter increased by >= $REQUIRED_DELTA"
  exit 0
else
  echo "[demo] FAIL: counter did not increase sufficiently"
  exit 1
fi


