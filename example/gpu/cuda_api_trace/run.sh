#!/bin/bash
set -u

ROOT_DIR=${BPFTIME_ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$ROOT_DIR"

export SPDLOG_LEVEL=info
export BPFTIME_LOG_LEVEL=INFO
export BPFTIME_LOG_OUTPUT=console
export BPFTIME_GLOBAL_SHM_NAME=bpftime_maps_shm_cuda_api
export BPFTIME_SHM_MEMORY_MB=256
export BPFTIME_CUDA_ROOT=${BPFTIME_CUDA_ROOT:-/usr/local/cuda}
export PATH="$BPFTIME_CUDA_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$ROOT_DIR/build/attach/text_segment_transformer:$BPFTIME_CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

RESULTS_DIR=${BPFTIME_RESULTS_DIR:-$ROOT_DIR/example/gpu/cuda_api_trace}
SERVER_LOG="$RESULTS_DIR/cuda-api-server.log"
CLIENT_LOG="$RESULTS_DIR/cuda-api-client.log"
: >"$SERVER_LOG"
: >"$CLIENT_LOG"

LD_PRELOAD="$ROOT_DIR/build/runtime/syscall-server/libbpftime-syscall-server.so" \
    "$ROOT_DIR/example/gpu/cuda_api_trace/cuda_api_trace" >"$SERVER_LOG" 2>&1 &
server_pid=$!
client_pid=""
cleanup() {
    if [ -n "$client_pid" ]; then
        kill -9 "$client_pid" 2>/dev/null || true
    fi
    kill -INT "$server_pid" 2>/dev/null || true
    sleep 1
    kill -9 "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 5
if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "CUDA API tracer exited before client launch"
    cat "$SERVER_LOG"
    exit 1
fi

timeout -s SIGKILL 30s env LD_PRELOAD="$ROOT_DIR/build/runtime/agent/libbpftime-agent.so" \
    "$ROOT_DIR/example/gpu/cuda_api_trace/vec_add_once" >"$CLIENT_LOG" 2>&1 &
client_pid=$!
wait "$client_pid"
client_rc=$?
client_pid=""

kill -INT "$server_pid" 2>/dev/null || true
sleep 2

echo "===== CUDA API tracer ====="
cat "$SERVER_LOG"
echo "===== CUDA client ====="
cat "$CLIENT_LOG"

if [ "$client_rc" -ne 0 ] && [ "$client_rc" -ne 124 ]; then
    echo "vec_add failed with exit code $client_rc"
    exit 1
fi
if ! grep -q "cudaLaunchKernel calls=" "$SERVER_LOG"; then
    echo "missing cudaLaunchKernel observations"
    exit 1
fi
if ! grep -q "vectorAdd passed: C\[0\]=0.0 C\[1\]=4.0" "$CLIENT_LOG"; then
    echo "missing correct vectorAdd output"
    exit 1
fi

echo "bpftime CUDA Runtime API trace passed"
