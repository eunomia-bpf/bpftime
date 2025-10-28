#!/usr/bin/env bash


set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build"

ENTRY_EXE="$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kprobe_entry/ptxpass_kprobe_entry"
RETPROBE_EXE="$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kretprobe/ptxpass_kretprobe"
MEMCAP_EXE="$BUILD_DIR/attach/nv_attach_impl/pass/ptxpass_kprobe_memcapture/ptxpass_kprobe_memcapture"

echo "[info] Using executables:\n  entry=$ENTRY_EXE\n  retprobe=$RETPROBE_EXE\n  memcapture=$MEMCAP_EXE"

test -x "$ENTRY_EXE" && test -x "$RETPROBE_EXE" && test -x "$MEMCAP_EXE"

# Ensure LLVM shared libs (if any) are discoverable
if command -v llvm-config >/dev/null 2>&1; then
  LLVM_LIBDIR=$(llvm-config --libdir 2>/dev/null || true)
  if [ -n "${LLVM_LIBDIR:-}" ]; then
    export LD_LIBRARY_PATH="$LLVM_LIBDIR:${LD_LIBRARY_PATH:-}"
  fi
fi

PTX_MIN='.version 7.0\n.target sm_60\n.visible .entry test() {\n  ret;\n}'

echo "[case] ptxpass_kprobe_entry --print-config has expected include/exclude"
OUT=$("$ENTRY_EXE" --print-config)
echo "$OUT" | grep -q "\^kprobe/.*\$"
echo "$OUT" | grep -q "\^kprobe/__memcapture\$"

echo "[case] ptxpass_kretprobe --config (no arg) prints defaults"
OUT=$("$RETPROBE_EXE" --config)
echo "$OUT" | grep -q "\^kretprobe/.*\$"

echo "[case] ptxpass_kprobe_memcapture --print-config has expected include"
OUT=$("$MEMCAP_EXE" --print-config)
echo "$OUT" | grep -q "\^kprobe/__memcapture\$"


echo "[case] entry pass matches kprobe/test with empty eBPF words"
JSON_INPUT=$(cat <<EOF
{"full_ptx":"$PTX_MIN","to_patch_kernel":"test","ebpf_instructions":[]}
EOF
)
PTX_ATTACH_POINT="kprobe/test" "$ENTRY_EXE" <<<"$JSON_INPUT" >/dev/null

echo "[case] retprobe pass unmatched attach point exits 0"
PTX_ATTACH_POINT="kprobe/test" "$RETPROBE_EXE" <<<"$JSON_INPUT" >/dev/null

echo "[case] retprobe pass matched attach point with empty eBPF should fail"
set +e
PTX_ATTACH_POINT="kretprobe/test" "$RETPROBE_EXE" <<<"$JSON_INPUT" >/dev/null
RC=$?
set -e
if [ "$RC" -eq 0 ]; then
  echo "[error] retprobe expected non-zero exit code on empty eBPF"
  exit 1
fi


echo "[case] memcapture pass matches kprobe/__memcapture with empty eBPF"
PTX_ATTACH_POINT="kprobe/__memcapture" "$MEMCAP_EXE" <<<"$JSON_INPUT" >/dev/null

echo "[info] PTX pass tests completed successfully."


