FROM ubuntu:24.04
WORKDIR /bpftime

# 更新软件包并安装依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev gcc g++ ca-certificates \
        clang-17 llvm-17 llvm-17-dev

# 复制源码
COPY . .

# 更新子模块
RUN git submodule update --init --recursive

# 设置环境变量
ENV CXX=g++
ENV CC=gcc
ENV LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
ENV PATH="${PATH}:/usr/lib/llvm-17/bin"

# 使用您提供的cmake配置进行构建
RUN cmake -Bbuild \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_BPFTIME_DAEMON=1 \
    -DCMAKE_C_COMPILER=/usr/lib/llvm-17/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/lib/llvm-17/bin/clang++ \
    -DLLVM_CONFIG=/usr/lib/llvm-17/bin/llvm-config \
    -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

# 编译
RUN cd build && make -j$(nproc)

# 安装到标准位置
RUN cd build && make install

# 添加到PATH
ENV PATH="${PATH}:/root/.bpftime/"