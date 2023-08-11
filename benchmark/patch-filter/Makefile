# BPF compiler
BPF_CC = clang
# BPF C flags
BPF_CFLAGS = -O2 -target bpf -c -g
# BPF source files
BPF_SRCS = $(wildcard *.bpf.c)
# BPF object files
BPF_OBJS = $(BPF_SRCS:.c=.o)

ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')
VMLINUX := ../../runtime/sdk/vmlinux/$(ARCH)/vmlinux.h
INCLUDES := -I../../runtime/sdk/include/ -I$(dir $(VMLINUX))

all: $(BPF_OBJS)

%.bpf.o: %.bpf.c
	$(BPF_CC) $(BPF_CFLAGS) $(INCLUDES) -D__TARGET_ARCH_$(ARCH) $< -o $@

vim.btf:
	pahole --btf_encode_detached vim.btf ./vim

clean:
	rm -f $(BPF_OBJS) base.btf
