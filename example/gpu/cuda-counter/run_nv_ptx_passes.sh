#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${BPFTIME_CUDA_ROOT:=/usr/local/cuda-12.6}"

echo "[info] Project root: ${PROJ_ROOT}"
echo "[info] Using BPFTIME_CUDA_ROOT=${BPFTIME_CUDA_ROOT}"

echo "[step] Configure & build bpftime (with CUDA attach)"
cmake -S "${PROJ_ROOT}" -B "${PROJ_ROOT}/build" -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT="${BPFTIME_CUDA_ROOT}" >/dev/null
cmake --build "${PROJ_ROOT}/build" --target bpftime_nv_attach_impl -j"$(nproc)" >/dev/null

echo "[step] Build CUDA example (vec_add + cuda_probe)"
make -C "${PROJ_ROOT}/example/gpu/cuda-counter" >/dev/null

SERVER_PRELOAD="${PROJ_ROOT}/build/runtime/syscall-server/libbpftime-syscall-server.so"
CLIENT_PRELOAD="${PROJ_ROOT}/build/runtime/agent/libbpftime-agent.so"

echo "[step] Launch eBPF server (cuda_probe)"
pushd "${PROJ_ROOT}" >/dev/null
BPFTIME_LOG_OUTPUT=console LD_PRELOAD="${SERVER_PRELOAD}" \
  "${PROJ_ROOT}/example/gpu/cuda-counter/cuda_probe" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true' EXIT
sleep 1

echo "[step] Run CUDA client (vec_add) with agent"
BPFTIME_LOG_OUTPUT=console LD_PRELOAD="${CLIENT_PRELOAD}" \
  "${PROJ_ROOT}/example/gpu/cuda-counter/vec_add" || true

echo "[check] Inspect transformed PTX and fatbin outputs"
if [[ -f /tmp/bpftime-recompile-nvcc/main.ptx ]]; then
  echo "- Found /tmp/bpftime-recompile-nvcc/main.ptx"
  echo "- Grep injected symbols (ptxpass_* regs, timer reads):"
  grep -nE '%ptxpass_(e0|r0|m0)|globaltimer' /tmp/bpftime-recompile-nvcc/main.ptx || true
else
  echo "- /tmp/bpftime-recompile-nvcc/main.ptx not found (pipeline may not have run)"
fi
if [[ -f /tmp/bpftime-recompile-nvcc/out.fatbin ]]; then
  echo "- Repacked fatbin: /tmp/bpftime-recompile-nvcc/out.fatbin (size)"
  stat -c '%n %s bytes' /tmp/bpftime-recompile-nvcc/out.fatbin || true
fi

echo "[done] Demo finished. Logs above show pipeline activity."
popd >/dev/null

