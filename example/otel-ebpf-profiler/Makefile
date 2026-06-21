# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
OUTPUT := .output
CLANG ?= clang
CC ?= gcc
GO ?= go

ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')
VMLINUX := ../../third_party/vmlinux/$(ARCH)/vmlinux.h
INCLUDES := -I../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX))

CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

BPF_OBJ := malloc_free.bpf.o
TRACER := otel_malloc_free

.PHONY: all tracer clean

all: $(BPF_OBJ) victim

tracer: $(TRACER)

$(BPF_OBJ): bpf/malloc_free.bpf.c $(VMLINUX)
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		$(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
		-c $< -o $@

$(TRACER): main.go go.mod go.sum $(BPF_OBJ)
	$(GO) build -o $@ .

victim: victim-src/victim.c
	$(CC) -g -O2 -Wall $< -o $@

clean:
	rm -rf $(OUTPUT) $(BPF_OBJ) $(TRACER) victim
