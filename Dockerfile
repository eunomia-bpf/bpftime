FROM debian:12

WORKDIR /bpftime

RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware \
    deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware \
    deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware \
    deb http://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/*
RUN apt-get update
RUN apt-get install -y autoconf make gcc g++ ftp wget gawk clang libncurses-dev libsm-dev libice-dev cmake pahole bpftool tmux vim nano pkg-config libelf-dev zlib1g-dev screen llvm-15 gdb

# Do them first, so docker can cache more things
COPY ./workloads/hotpatch-poc/end-to-end-poc/update-perl-image.sh /bpftime/
RUN /bin/bash /bpftime/update-perl-image.sh
RUN cpan install Text::Template

COPY ./workloads/hotpatch-poc/end-to-end-poc/assets /bpftime/assets
RUN cd /bpftime/assets && tar -zxvf automake-1.14.tar.gz && \
    cd automake-1.14 && \
    ./configure --prefix=/usr/local/automake-1.14 && \
    make -j install && \
    ln -s /usr/local/automake-1.14/bin/aclocal-1.14 /usr/bin/aclocal-1.14 && \
    ln -s /usr/local/automake-1.14/bin/automake-1.14 /usr/bin/automake-1.14 

RUN cd /bpftime/assets && tar -zxvf automake-1.15.tar.gz && \
    cd automake-1.15 && \
    ./configure --prefix=/usr/local/automake-1.15 && \
    make -j install && \
    ln -s /usr/local/automake-1.15/bin/aclocal-1.15 /usr/bin/aclocal-1.15 && \
    ln -s /usr/local/automake-1.15/bin/automake-1.15 /usr/bin/automake-1.15 

# Also build them before we copy the whole folder. Cache more things.
RUN mkdir -p /bpftime/workloads/hotpatch-poc/
COPY ./workloads/hotpatch-poc/ /bpftime/workloads/hotpatch-poc/
RUN /bin/bash /bpftime/workloads/hotpatch-poc/end-to-end-poc/scripts/build_libevent.sh

# Build the whole project
COPY . /bpftime/temp/
# but remove workloads/hotpatch-poc, since we already copied them
RUN rm -rf /bpftime/temp/workloads/hotpatch-poc
RUN cp -r /bpftime/temp/* /bpftime && rm -rf /bpftime/temp
RUN rm -rf /bpftime/build /bpftime/.cache /bpftime/assets
RUN /bin/bash /bpftime/build-all.sh

ENV BPFTIME_DISABLE_JIT=1

# Build programs for poc1, 2, 3
RUN cd /bpftime/workloads/ebpf-patch-dev/poc1-libevent && bash build_patch.sh
RUN cd /bpftime/workloads/ebpf-patch-dev/poc2-libevent && bash build_patch.sh
RUN make -C /bpftime/workloads/ebpf-patch-dev/poc3-openssl

# build for poc4 and poc5
RUN make -C /bpftime/workloads/ebpf-patch-dev/poc4-redis
RUN make -C /bpftime/workloads/ebpf-patch-dev/poc5-vim

# Convenient for working with fixed paths..
RUN mkdir -p /work/bpf-dev/patch-dev && ln -s /bpftime /work/bpf-dev/patch-dev/bpftime
RUN mkdir -p /home/yunwei && ln -s /bpftime /home/yunwei/bpftime

