# Platform defaults when targeting QNX (Neutrino 8.0+)
#
# Included from cmake/StandardSettings.cmake when BPFTIME_TARGET_QNX=ON
# or when CMAKE_SYSTEM_NAME is QNX.

if(NOT BPFTIME_TARGET_QNX AND CMAKE_SYSTEM_NAME STREQUAL "QNX")
  set(BPFTIME_TARGET_QNX ON CACHE BOOL "Build for QNX Neutrino" FORCE)
endif()

if(NOT BPFTIME_TARGET_QNX)
  return()
endif()

message(STATUS "Applying QNX platform defaults (phase-1: process hook / uprobe)")

# Phase-1: userspace VM + Frida uprobe only
set(BPFTIME_BUILD_WITH_LIBBPF OFF CACHE BOOL "libbpf not available on QNX" FORCE)
set(BPFTIME_BUILD_KERNEL_BPF OFF CACHE BOOL "No kernel eBPF on QNX" FORCE)
set(BUILD_BPFTIME_DAEMON OFF CACHE BOOL "Daemon requires Linux eBPF" FORCE)
set(BPFTIME_ENABLE_CUDA_ATTACH OFF CACHE BOOL "No CUDA attach on QNX" FORCE)
set(BPFTIME_ENABLE_IOURING_EXT OFF CACHE BOOL "io_uring is Linux-only" FORCE)
set(BPFTIME_ENABLE_UNIT_TESTING OFF CACHE BOOL "Disable host unit tests for QNX cross-build" FORCE)
set(BPFTIME_BUILD_STATIC_LIB OFF CACHE BOOL "Static archive packing uses Linux ar scripts" FORCE)
set(ENABLE_EBPF_VERIFIER OFF CACHE BOOL "Verifier not in phase-1 QNX scope" FORCE)
set(BUILD_ATTACH_IMPL_EXAMPLE OFF CACHE BOOL "" FORCE)

# Prefer uBPF JIT for phase-1 to avoid LLVM cross-compile; enable LLVM later.
if(NOT DEFINED BPFTIME_QNX_FORCE_LLVM)
  set(BPFTIME_LLVM_JIT OFF CACHE BOOL "Disable LLVM JIT for QNX phase-1 (set BPFTIME_QNX_FORCE_LLVM=ON to override)" FORCE)
  set(BPFTIME_UBPF_JIT ON CACHE BOOL "Use uBPF JIT on QNX phase-1" FORCE)
endif()

add_compile_definitions(BPFTIME_TARGET_QNX=1)
# Some code paths check __QNX__; qcc usually defines it, reinforce for other compilers.
add_compile_definitions(__QNX__=1)

# QNX typically has no ncurses; avoid unconditional link from root CMakeLists.
set(BPFTIME_QNX_SKIP_NCURSES ON CACHE BOOL "" FORCE)

message(STATUS "  BPFTIME_BUILD_WITH_LIBBPF=${BPFTIME_BUILD_WITH_LIBBPF}")
message(STATUS "  BUILD_BPFTIME_DAEMON=${BUILD_BPFTIME_DAEMON}")
message(STATUS "  BPFTIME_LLVM_JIT=${BPFTIME_LLVM_JIT}")
message(STATUS "  BPFTIME_UBPF_JIT=${BPFTIME_UBPF_JIT}")
