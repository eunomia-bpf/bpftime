#! /bin/bash

set -e

# build the no mpk version
cmake -Bbuild -DLLVM_DIR=/usr/lib/llvm-15/cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBPFTIME_LLVM_JIT=1 \
    -DBPFTIME_ENABLE_LTO=1 \
    -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO \
    -DENABLE_PROBE_WRITE_CHECK=0 \
    -DENABLE_PROBE_READ_CHECK=0 \
    -DBPFTIME_ENABLE_MPK=0
cmake --build build --config RelWithDebInfo --target install -j

# build the mpk version
cmake -Bbuild-mpk -DLLVM_DIR=/usr/lib/llvm-15/cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBPFTIME_LLVM_JIT=1 \
    -DBPFTIME_ENABLE_LTO=1 \
    -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO \
    -DENABLE_PROBE_WRITE_CHECK=0 \
    -DENABLE_PROBE_READ_CHECK=0 \
    -DBPFTIME_ENABLE_MPK=1
cmake --build build-mpk --config RelWithDebInfo --target install -j
