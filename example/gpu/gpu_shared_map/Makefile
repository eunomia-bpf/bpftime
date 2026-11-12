# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
OUTPUT := .output
CLANG ?= clang
LIBBPF_SRC := $(abspath ../../../third_party/libbpf/src)
BPFTOOL_SRC := $(abspath ../../../third_party/bpftool/src)
LIBBPF_OBJ := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(OUTPUT)/bpftool)
BPFTOOL ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')
VMLINUX := ../../../third_party/vmlinux/$(ARCH)/vmlinux.h
INCLUDES := -I$(OUTPUT) -I../../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX))
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

APPS = gpu_shared_map

.PHONY: all clean
all: $(APPS) vec_add

vec_add: vec_add.cu
	@if command -v nvcc >/dev/null 2>&1; then \
		nvcc -cudart shared vec_add.cu -o vec_add -g; \
	else \
		echo "Warning: CUDA not found, skipping vec_add build"; \
	fi

clean:
	rm -rf $(OUTPUT) $(APPS) vec_add

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	mkdir -p $@

$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	make -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1 \
		OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@) \
		INCLUDEDIR= LIBDIR= UAPIDIR= \
		install

$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	make ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap

# Clang 系统头补全（与其它示例保持一致）
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(CLANG) -Xlinker --export-dynamic -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		 $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
		 -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL)
	$(BPFTOOL) gen skeleton $< > $@

# 确保主程序对象在编译前生成 skeleton
$(patsubst %,$(OUTPUT)/%.o,$(APPS)): %.o: %.skel.h

$(OUTPUT)/%.o: %.c $(wildcard %.h) | $(OUTPUT)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

$(APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

.DELETE_ON_ERROR:
.SECONDARY:

