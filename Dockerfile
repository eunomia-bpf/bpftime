FROM ubuntu:24.04
WORKDIR /bpftime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev gcc g++ ca-certificates \
        clang-17 llvm-17 llvm-17-dev

COPY . .

RUN git submodule update --init --recursive

ENV BPFTIME_VM_NAME=llvm 
ENV LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
ENV PATH="${PATH}:/usr/lib/llvm-17/bin"

RUN rm -rf build && mkdir build && cmake -Bbuild \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_BPFTIME_DAEMON=1 \
    -DCMAKE_C_COMPILER=/usr/lib/llvm-17/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/lib/llvm-17/bin/clang++ \
    -DLLVM_CONFIG=/usr/lib/llvm-17/bin/llvm-config \
    -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

RUN cd build && make -j$(nproc)
RUN cd build && make install
ENV PATH="${PATH}:/root/.bpftime/"