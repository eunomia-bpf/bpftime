#!/bin/sh
# Run a test binary. If it dies on a signal, print every thread's backtrace
# from its own core dump, so an intermittent crash on a CI runner leaves a
# stack behind instead of only "SIGSEGV" at a TEST_CASE line.
#
# Usage: run_with_backtrace.sh <binary> [args...]
#
# Some test cases crash child processes on purpose; their cores are ignored:
# only the core of the binary started here (matched by its pid) is read.
# Needs a privileged job container to set kernel.core_pattern; without it the
# binary still runs and its exit status is kept, just with no backtrace.
set -u
cores=$(mktemp -d)
if ! { echo "$cores/core.%p" > /proc/sys/kernel/core_pattern; } 2>/dev/null; then
  echo "run_with_backtrace: cannot set kernel.core_pattern; no backtrace on a crash"
fi
ulimit -c unlimited 2>/dev/null || true

"$@" &
pid=$!
rc=0
wait "$pid" || rc=$?

if [ "$rc" -gt 128 ] && [ -f "$cores/core.$pid" ]; then
  echo "::group::$1 died on signal $((rc - 128)): backtraces of every thread"
  if ! command -v gdb >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      apt-get install -y -q gdb >/dev/null 2>&1 ||
        { apt-get update -q >/dev/null 2>&1 && apt-get install -y -q gdb >/dev/null 2>&1; }
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y -q gdb >/dev/null 2>&1
    fi
  fi
  if command -v gdb >/dev/null 2>&1; then
    gdb -q -batch -ex "info threads" -ex "thread apply all bt" "$1" "$cores/core.$pid"
  else
    echo "run_with_backtrace: gdb is not available; core kept at $cores/core.$pid"
  fi
  echo "::endgroup::"
fi
exit "$rc"
