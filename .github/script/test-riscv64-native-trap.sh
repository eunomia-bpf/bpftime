#!/usr/bin/env bash
set -euo pipefail

case "$(uname -m)" in
    riscv64) ;;
    *)
        echo "riscv64 host required; refusing qemu-user or cross-architecture evidence" >&2
        exit 2
        ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=no --ignore-submodules=none)" ]]; then
    echo "clean tracked worktree and submodules required for reproducible evidence" >&2
    exit 3
fi

build_dir="${1:-build-native-riscv64-trap}"
build_type="${BUILD_TYPE:-Release}"

echo "commit=$(git rev-parse HEAD)"
echo "kernel=$(uname -srvm)"
lscpu
echo "hardware provenance must be supplied separately by the operator"

cmake -S . -B "$build_dir" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DBUILD_BPFTIME_DAEMON=OFF \
    -DBPFTIME_ENABLE_UNIT_TESTING=ON \
    -DBPFTIME_ENABLE_FRIDA=OFF \
    -DBPFTIME_ENABLE_TRAP_UPROBE=ON \
    -DENABLE_EBPF_VERIFIER=OFF \
    -DBPFTIME_LLVM_JIT=OFF \
    -DBPFTIME_BUILD_WITH_LIBBPF=OFF

cmake --build "$build_dir" \
    --target bpftime_trap_uprobe_attach_tests bpftime_trap_late_host \
    -j"$(nproc)"

trap_tests="$build_dir/attach/trap_uprobe_attach_impl/bpftime_trap_uprobe_attach_tests"
late_host="$build_dir/attach/trap_uprobe_attach_impl/bpftime_trap_late_host"
late_module="$build_dir/attach/trap_uprobe_attach_impl/libbpftime_trap_late_module.so"

"$trap_tests" "Trap backend: concurrent three-phase patch*"
"$trap_tests" "Trap backend: handler signal-safety*,Trap backend: new threads*"
"$late_host" "$late_module"
