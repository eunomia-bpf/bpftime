#!/usr/bin/env bash
# Runs one GPU example the way its README describes (server and victim,
# direct run, or late attach) and checks its output. Called by
# run_gpu_examples.py for each example from the repository root, with
# GPU_EXAMPLE_NAME, GPU_EXAMPLE_PATH, GPU_EXAMPLE_EXECUTABLE,
# GPU_EXAMPLE_VICTIM, GPU_EXAMPLE_EXPECTED, GPU_EXAMPLE_MODE and
# GPU_EXAMPLE_VERIFIER_LEVEL set and
# bpftime built in build/. Writes server.log, client.log and tool.log in
# the repository root.
set -euo pipefail
ROOT_DIR=$(pwd)
EXAMPLE_DIR="$ROOT_DIR/example/gpu/${GPU_EXAMPLE_PATH}"
BUILD_DIR="$ROOT_DIR/build"
SERVER_SO="$BUILD_DIR/runtime/syscall-server/libbpftime-syscall-server.so"
AGENT_SO="$BUILD_DIR/runtime/agent/libbpftime-agent.so"
EXE_REL="${GPU_EXAMPLE_EXECUTABLE}"
VICTIM_REL="${GPU_EXAMPLE_VICTIM}"
EXPECTED="${GPU_EXAMPLE_EXPECTED}"
EXAMPLE_NAME="${GPU_EXAMPLE_NAME}"
EXAMPLE_MODE="${GPU_EXAMPLE_MODE}"
VERIFIER_LEVEL="${GPU_EXAMPLE_VERIFIER_LEVEL:-STRICT}"
CLIENT_TIMEOUT=30
BPFTIME_CLI="$BUILD_DIR/tools/cli/bpftime"
BPFTIMETOOL="$BUILD_DIR/tools/bpftimetool/bpftimetool"
export SPDLOG_LEVEL=info
export BPFTIME_LOG_LEVEL=INFO
export BPFTIME_VERIFIER_LEVEL="$VERIFIER_LEVEL"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export BPFTIME_GLOBAL_SHM_NAME="bpftime_maps_shm_${EXAMPLE_NAME}"
# GPU per-thread map sizing (some examples are memory-sensitive).
GPU_THREAD_COUNT=8192
case "$EXAMPLE_NAME" in
  kernelretsnoop) GPU_THREAD_COUNT=10 ;;       # per README: avoid bad_alloc
  threadhist) GPU_THREAD_COUNT=524288 ;;       # per README: allow large thread id space
  kernel_trace) GPU_THREAD_COUNT=4096 ;;       # kernel_trace vec_add launches 32*128=4096 threads
esac
export BPFTIME_MAP_GPU_THREAD_COUNT="$GPU_THREAD_COUNT"
export LD_LIBRARY_PATH="$ROOT_DIR/build/attach/text_segment_transformer:${LD_LIBRARY_PATH:-}"
# The PTX pass/compiler libraries are baked into ptx_pass_config.h with
# the build runner's absolute workspace path, which does not exist on
# the containerized test runners. Point both loaders at the artifacts
# extracted under this job's workspace instead.
export BPFTIME_PTXPASS_LIBRARIES="$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kprobe_entry/libptxpass_kprobe_entry.so:$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kretprobe/libptxpass_kretprobe.so:$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kprobe_memcapture/libptxpass_kprobe_memcapture.so"
export BPFTIME_PTX_COMPILER_LIB="$BUILD_DIR/attach/nv_attach_impl/ptx_compiler/libnv_attach_impl_ptx_compiler.so"
df -h /dev/shm || true
# Best-effort cleanup in case a previous job was interrupted.
# Note: `pkill -f` matches argv only (not environment variables),
# so we must kill known binaries explicitly.
for proc in atomizer kernel_trace gpu_shared_map threadscheduling cuda_probe threadhist directly_run kernelretsnoop launchlate mem_trace vec_add vec_add_graph cutlass_launch_counter cutlass_gemm; do
  pkill -9 -x "$proc" 2>/dev/null || true
done
SERVER_BIN="$(basename "${EXE_REL%% *}")"
pkill -9 -x "$SERVER_BIN" 2>/dev/null || true
if [ -n "${VICTIM_REL:-}" ]; then
  VICTIM_BIN="$(basename "${VICTIM_REL%% *}")"
  pkill -9 -x "$VICTIM_BIN" 2>/dev/null || true
fi
# Remove bpftime shared memory segments and semaphores
rm -f /dev/shm/bpftime* || true
rm -f /dev/shm/sem.bpftime* || true
# Re-check after cleanup: lingering processes can keep shm pinned.
df -h /dev/shm || true
# ensure previous logs do not interfere
rm -f server.log client.log tool.log || true
: > server.log
: > client.log
: > tool.log
# tune shared memory size per example (ringbuf-heavy examples need more)
# 16MB is often too small once CUDA init + maps are involved.
SHM_MB=64
case "$EXAMPLE_NAME" in
  kernelretsnoop) SHM_MB=64 ;;  # ringbuf map needs larger shm
  kernel_trace) SHM_MB=256 ;;
  threadhist) SHM_MB=64 ;;
  threadscheduling) SHM_MB=64 ;;
  atomizer) SHM_MB=64 ;;
  directly_run_on_gpu) SHM_MB=64 ;;
esac
# Guard against small /dev/shm (common in containerized runners).
# If /dev/shm is smaller than requested, shared_memory_object::truncate()
# will fail with ENOSPC ("No space left on device").
AVAIL_MB=$(df --output=avail -m /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
if [[ "${AVAIL_MB:-}" =~ ^[0-9]+$ ]] && [ "$AVAIL_MB" -gt 0 ]; then
  HEADROOM_MB=8
  MAX_MB=$((AVAIL_MB - HEADROOM_MB))
  if [ "$MAX_MB" -lt 32 ]; then
    echo "Not enough /dev/shm available: avail=${AVAIL_MB}MB (need at least 32MB)."
    df -h /dev/shm || true
    ls -lh /dev/shm | head -n 200 || true
    exit 1
  fi
  if [ "$SHM_MB" -gt "$MAX_MB" ]; then
    echo "Clamping BPFTIME_SHM_MEMORY_MB from $SHM_MB to $MAX_MB due to /dev/shm avail=${AVAIL_MB}MB"
    SHM_MB="$MAX_MB"
  fi
fi
# Default to "not an intentional teardown"; each victim path sets
# this to 1 only after confirming the process was alive and reaped
# by our own timeout/teardown. Initialized before the mode branches
# so the validation gate never sees a stale or post-branch reset.
CLIENT_KILLED_INTENTIONALLY=0
if [ "$EXAMPLE_MODE" = "direct-run" ]; then
  pushd "$EXAMPLE_DIR" >/dev/null
  echo "Launching direct-run server with $SERVER_SO: $EXE_REL (BPFTIME_SHM_MEMORY_MB=$SHM_MB)"
  # Don't preload into the shell itself; only preload into the target binary.
  BPFTIME_SHM_MEMORY_MB=$SHM_MB BPFTIME_LOG_OUTPUT=console bash -lc "LD_PRELOAD=\"$SERVER_SO\" exec $EXE_REL" > "$ROOT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  popd >/dev/null
  echo "Server PID: $SERVER_PID"
  sleep 15
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server crashed early. Showing server.log tail:"; tail -n 200 server.log || true
    exit 1
  fi
  if [ ! -x "$BPFTIMETOOL" ]; then
    echo "bpftimetool not found at $BPFTIMETOOL"; tail -n 200 server.log || true; exit 1
  fi
  echo "Running bpftimetool run-on-cuda (GEMM) ..."
  set +e
  timeout -s SIGKILL ${CLIENT_TIMEOUT}s "$BPFTIMETOOL" run-on-cuda cuda__gemm 1 4 4 1 16 16 1 > tool.log 2>&1
  TOOL_RC=$?
  set -e
  sleep 5
elif [ "$EXAMPLE_MODE" = "dynamic-attach" ]; then
  if [ ! -x "$BPFTIME_CLI" ]; then
    echo "bpftime CLI not found at $BPFTIME_CLI"
    echo "--- build/tools/cli (ls) ---"
    ls -la "$BUILD_DIR/tools/cli" || true
    exit 1
  fi
  pushd "$EXAMPLE_DIR" >/dev/null
  echo "Launching target (no preload): $VICTIM_REL"
  # Some CI environments restrict ptrace/Frida injection unless the
  # target explicitly allows it (e.g., yama ptrace_scope=1). The
  # dynamic_hook victim can opt-in via BPFTIME_ALLOW_PTRACE.
  env BPFTIME_ALLOW_PTRACE=1 bash -lc "exec $VICTIM_REL" > "$ROOT_DIR/client.log" 2>&1 &
  VICTIM_PID=$!
  popd >/dev/null
  echo "Victim PID: $VICTIM_PID"
  sleep 5
  if ! kill -0 "$VICTIM_PID" 2>/dev/null; then
    echo "Victim crashed early. Showing client.log tail:"; tail -n 200 client.log || true
    exit 1
  fi
  pushd "$EXAMPLE_DIR" >/dev/null
  echo "Launching bpftime trace (late attach): $EXE_REL (target pid=$VICTIM_PID)"
  env BPFTIME_SHM_MEMORY_MB=$SHM_MB BPFTIME_LOG_OUTPUT=console \
    "$BPFTIME_CLI" trace --pid "$VICTIM_PID" --auto-refresh-ms 500 \
    $EXE_REL > "$ROOT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  popd >/dev/null
  echo "Trace PID: $SERVER_PID"
  sleep 15
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    # Let trace handle detach/cleanup.
    kill -INT "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # Only treat the victim's 137 as the intended teardown if it was
  # still alive when we killed it. A process that already died (for
  # example an OOM SIGKILL) must not be excused just because 137
  # matches the teardown code.
  set +e
  if kill -0 "$VICTIM_PID" 2>/dev/null; then
    CLIENT_KILLED_INTENTIONALLY=1
    kill -9 "$VICTIM_PID" 2>/dev/null || true
  else
    CLIENT_KILLED_INTENTIONALLY=0
  fi
  grace=0
  while [ "$grace" -lt 10 ] &&
        kill -0 "$VICTIM_PID" 2>/dev/null; do
    sleep 1
    grace=$((grace + 1))
  done
  if ! kill -0 "$VICTIM_PID" 2>/dev/null; then
    wait "$VICTIM_PID" 2>/dev/null
    CLIENT_RC=$?
  else
    echo "FAILURE: victim still alive after teardown grace period"
    CLIENT_RC=1
  fi
  set -e
else
  pushd "$EXAMPLE_DIR" >/dev/null
  echo "Launching server with $SERVER_SO: $EXE_REL (BPFTIME_SHM_MEMORY_MB=$SHM_MB)"
  # Don't preload into the shell itself; only preload into the target binary.
  BPFTIME_SHM_MEMORY_MB=$SHM_MB BPFTIME_LOG_OUTPUT=console bash -lc "LD_PRELOAD=\"$SERVER_SO\" exec $EXE_REL" > "$ROOT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  popd >/dev/null
  echo "Server PID: $SERVER_PID"
  sleep 5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server crashed early. Showing server.log tail:"; tail -n 200 server.log || true
    exit 1
  fi
  echo "Launching client with $AGENT_SO: $VICTIM_REL"
  pushd "$EXAMPLE_DIR" >/dev/null
  set +e
  # Don't preload into the shell itself; only preload into the target
  # binary. Run it in the background and bound the window ourselves so
  # an exit-137 caused by an external OOM kill (before the window
  # elapses) is distinguishable from our own timeout teardown.
  env BPFTIME_LOG_OUTPUT=console bash -lc "LD_PRELOAD=\"$AGENT_SO\" exec $VICTIM_REL" > "$ROOT_DIR/client.log" 2>&1 &
  CLIENT_PID=$!
  CLIENT_KILLED_INTENTIONALLY=0
  for ((elapsed = 0; elapsed < CLIENT_TIMEOUT; ++elapsed)); do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$CLIENT_PID" 2>/dev/null; then
    # Still running after the bounded window: intended teardown.
    kill -9 "$CLIENT_PID" 2>/dev/null || true
    CLIENT_KILLED_INTENTIONALLY=1
  fi
  # Reap with a script-level bound: only wait once the process is
  # observed dead, so an uninterruptible victim cannot stall the job
  # past the teardown window. If it survives the grace period the
  # SIGKILL did not take effect, which is a real failure rather than
  # an intended teardown — do not fabricate a passing 137.
  grace=0
  while [ "$grace" -lt 10 ] &&
        kill -0 "$CLIENT_PID" 2>/dev/null; do
    sleep 1
    grace=$((grace + 1))
  done
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    wait "$CLIENT_PID" 2>/dev/null
    CLIENT_RC=$?
  else
    echo "FAILURE: client still alive after teardown grace period"
    CLIENT_RC=1
  fi
  set -e
  popd >/dev/null
fi
echo "Checking expected output: $EXPECTED"
# Aggregate failures into RESULT instead of aborting under set -e:
# the background server still needs cleanup below and all three logs
# are uploaded as artifacts. Start green; every check below only
# flips RESULT to 1, so a later check can never mask an earlier
# failure.
RESULT=0
# TOOL_RC is set only on the direct-run path: bpftimetool
# run-on-cuda is a finite workload wrapped in timeout, so any
# non-zero status - including 137 from a timeout SIGKILL on a hang
# - is a real failure even if tool.log contains the expected text.
if [ "${TOOL_RC:-0}" -ne 0 ]; then
  echo "FAILURE: bpftimetool run-on-cuda exited with code ${TOOL_RC}"; tail -n 200 tool.log || true
  RESULT=1
fi
# CLIENT_RC is set on both victim paths. A 137 is accepted only when
# we verified the process was still alive and killed it as the
# intended timeout/teardown (CLIENT_KILLED_INTENTIONALLY=1). A 137
# from an external OOM kill that happened on its own, or any other
# non-zero status, is a real failure and fails the case.
if [ -n "${CLIENT_RC:-}" ] && [ "$CLIENT_RC" -ne 0 ] &&
   { [ "$CLIENT_RC" -ne 137 ] ||
     [ "${CLIENT_KILLED_INTENTIONALLY:-0}" -ne 1 ]; }; then
  echo "FAILURE: victim/client exited unexpectedly with code $CLIENT_RC (intentional_kill=${CLIENT_KILLED_INTENTIONALLY:-0})"; tail -n 200 client.log || true
  RESULT=1
fi
# treat crashes as failure even if expected string appears
if grep -Eiq "(core dumped|Segmentation fault|Aborted)" server.log client.log tool.log; then
  echo "Detected crash in logs"; tail -n 200 server.log || true; tail -n 200 client.log || true; tail -n 200 tool.log || true; RESULT=1
else
  EXAMPLE_NAME="${GPU_EXAMPLE_NAME}"
  FOUND=1
  case "$EXAMPLE_NAME" in
    directly_run_on_gpu)
      # For GEMM, expect a non-zero fixed-point value after bpftimetool ran.
      if grep -Eq "GEMM C\\[0\\]=[1-9]" server.log; then
        FOUND=0
      fi
      ;;
    cutlass)
      # Counter prints periodically; require a non-zero count to ensure attachment worked.
      if grep -Eq "CUTLASS launches: [1-9]" server.log; then
        FOUND=0
      fi
      ;;
    cuda-counter)
      if grep -Fq "calls:" server.log || grep -Fq "C[0] =" client.log || grep -Fq "C[1] =" client.log; then
        FOUND=0
      fi
      ;;
    mem_trace)
      if grep -Fq "counter[0]=" server.log || grep -Fq "mem_traces:" server.log; then
        FOUND=0
      fi
      ;;
    *)
      if grep -Fq "$EXPECTED" server.log client.log; then
        FOUND=0
      fi
      ;;
  esac
  if [ $FOUND -eq 0 ]; then
    echo "SUCCESS: found README-like output for $EXAMPLE_NAME"
    # Keep RESULT untouched: an exit-code failure above must not be
    # overwritten by a matching expected string.
  else
    echo "FAILURE: expected output not found for $EXAMPLE_NAME"
    echo "--- server.log (tail) ---"; tail -n 200 server.log || true
    echo "--- client.log (tail) ---"; tail -n 200 client.log || true
    echo "--- tool.log (tail) ---"; tail -n 200 tool.log || true
    RESULT=1
  fi
fi
if [ -n "${SERVER_PID:-}" ]; then
  kill -9 $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
fi
exit $RESULT
