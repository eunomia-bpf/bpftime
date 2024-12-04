FROM ubuntu:24.04
WORKDIR /bpftime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev gcc g++ ca-certificates clang llvm
COPY . .
RUN git submodule update --init --recursive
ENV CXX=g++
ENV CC=gcc
RUN make release
ENV PATH="${PATH}:/root/.bpftime/"
