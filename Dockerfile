FROM ubuntu:23.04
WORKDIR /bpftime
RUN apt-get update && apt install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make cmake git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev  gcc-12 g++-12 ca-certificates clang
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
COPY . .
RUN git submodule update --init --recursive
ENV CXX=g++-12
ENV CC=gcc-12
RUN make release

