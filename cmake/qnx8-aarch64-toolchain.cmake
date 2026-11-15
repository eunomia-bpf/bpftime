# QNX Software Development Platform 8.0 — aarch64 (little-endian) cross toolchain
#
# Usage:
#   source /path/to/qnxsdp-env.sh   # sets QNX_HOST / QNX_TARGET
#   cmake -B build-qnx \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/qnx8-aarch64-toolchain.cmake \
#     -DBPFTIME_TARGET_QNX=ON ...
#
# Expected environment:
#   QNX_HOST   – host tools (qcc, q++, ntoaarch64-ar, ...)
#   QNX_TARGET – target sysroot for aarch64le

if(NOT DEFINED ENV{QNX_HOST} OR NOT DEFINED ENV{QNX_TARGET})
  message(FATAL_ERROR
    "QNX_HOST and QNX_TARGET must be set. Source the QNX SDP 8.0 environment script first.")
endif()

set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_VERSION 8.0)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "${QNX_HOST}/usr/bin/qcc" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${QNX_HOST}/usr/bin/q++" CACHE FILEPATH "" FORCE)
set(CMAKE_AR "${QNX_HOST}/usr/bin/ntoaarch64-ar" CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${QNX_HOST}/usr/bin/ntoaarch64-ranlib" CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP "${QNX_HOST}/usr/bin/ntoaarch64-strip" CACHE FILEPATH "" FORCE)
set(CMAKE_NM "${QNX_HOST}/usr/bin/ntoaarch64-nm" CACHE FILEPATH "" FORCE)

# QNX qcc/q++ architecture selector for aarch64 little-endian
set(QNX_ARCH_FLAG "-Vgcc_ntoaarch64le")
set(CMAKE_C_FLAGS_INIT "${QNX_ARCH_FLAG}")
set(CMAKE_CXX_FLAGS_INIT "${QNX_ARCH_FLAG} -lang-c++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${QNX_ARCH_FLAG}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${QNX_ARCH_FLAG}")

set(CMAKE_FIND_ROOT_PATH "${QNX_TARGET}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Shared libraries use .so on QNX
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")

# Mark that we are cross-compiling for QNX
set(BPFTIME_CROSS_COMPILING_QNX TRUE CACHE BOOL "Building for QNX via this toolchain" FORCE)

message(STATUS "QNX toolchain: HOST=${QNX_HOST}")
message(STATUS "QNX toolchain: TARGET=${QNX_TARGET}")
message(STATUS "QNX toolchain: arch=aarch64le (qcc ${QNX_ARCH_FLAG})")
