#!/usr/bin/env bash
# Phase-1 QNX 8.0 aarch64 build helper for bpftime (process hook / uprobe).
#
# Prerequisites:
#   1. Source QNX SDP 8.0 env (sets QNX_HOST / QNX_TARGET)
#   2. Build Frida for qnx-arm64 and set BPFTIME_FRIDA_QNX_ROOT
#   3. Boost.Interprocess available for the QNX target sysroot
#
# Usage:
#   export BPFTIME_FRIDA_QNX_ROOT=/path/to/frida-qnx-arm64-out
#   ./scripts/qnx/build-phase1.sh [build-dir]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-${ROOT}/build-qnx}"

if [[ -z "${QNX_HOST:-}" || -z "${QNX_TARGET:-}" ]]; then
  echo "ERROR: QNX_HOST and QNX_TARGET must be set (source qnxsdp-env.sh)" >&2
  exit 1
fi

if [[ -z "${BPFTIME_FRIDA_QNX_ROOT:-}" ]]; then
  echo "ERROR: BPFTIME_FRIDA_QNX_ROOT must point to Frida qnx-arm64 output" >&2
  echo "  expected: \$BPFTIME_FRIDA_QNX_ROOT/{gum,core}/libfrida-*.a" >&2
  exit 1
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

cmake -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/cmake/qnx8-aarch64-toolchain.cmake" \
  -DBPFTIME_TARGET_QNX=ON \
  -DBPFTIME_BUILD_WITH_LIBBPF=OFF \
  -DBUILD_BPFTIME_DAEMON=OFF \
  -DBPFTIME_LLVM_JIT=OFF \
  -DBPFTIME_UBPF_JIT=ON \
  -DBPFTIME_FRIDA_QNX_ROOT="${BPFTIME_FRIDA_QNX_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  "${ROOT}"

cmake --build "${BUILD_DIR}" \
  --target bpftime-agent bpftime-cli-cpp bpftimetool \
  -j"${JOBS}"

echo ""
echo "Phase-1 QNX build finished."
echo "  agent:  ${BUILD_DIR}/runtime/agent/libbpftime-agent.so"
echo "  cli:    ${BUILD_DIR}/tools/cli/bpftime"
echo "  tool:   ${BUILD_DIR}/tools/bpftimetool/bpftimetool"
echo ""
echo "See scripts/qnx/README.md for phase-1 scope and remaining work."
