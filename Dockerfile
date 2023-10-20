FROM ubuntu:23.04
WORKDIR /bpftime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev  gcc g++ ca-certificates clang llvm
RUN apt-get install -y --no-install-recommends curl && \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
COPY . .
RUN git submodule update --init --recursive
ENV CXX=g++
ENV CC=gcc
ENV PATH=${PATH}:/root/.cargo/bin
RUN make release && make install
