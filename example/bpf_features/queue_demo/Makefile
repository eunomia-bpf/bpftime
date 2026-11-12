# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
OUTPUT := .output
CLANG := clang
LLVM_STRIP := llvm-strip
BPFTOOL := bpftool
MAKE := make
LIBBPF_SRC := $(abspath ../../../third_party/libbpf/src)
BPFTOOL_SRC := $(abspath ../../../third_party/bpftool/src)
LIBBPF_OBJ := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT := $(abspath $(OUTPUT)/bpftool)
BPFTOOL_EXEC := $(BPFTOOL_OUTPUT)/bootstrap/bpftool
ARCH := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/' | sed 's/ppc64le/powerpc/' | sed 's/mips.*/mips/')
VMLINUX := ../../../third_party/vmlinux/$(ARCH)/vmlinux.h
# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated
INCLUDES := -I$(OUTPUT) -I../../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX))
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

APPS = uprobe_queue uprobe_stack target

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architectures/distros don't have /usr/include/asm/types.h included for deps
CLANG_BPF_SYS_INCLUDES = $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-I\1|p }')

ifeq ($(wildcard $(VMLINUX)),)
$(error ERROR: missing $(VMLINUX))
endif

.PHONY: all
all: $(APPS)

.PHONY: clean
clean:
	$(RM) -r $(OUTPUT) $(APPS)

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	mkdir -p $@

# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1		      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@)		      \
		    INCLUDEDIR= LIBDIR= UAPIDIR=			      \
		    install

# Build bpftool
$(BPFTOOL_EXEC): | $(BPFTOOL_OUTPUT)
	$(MAKE) -C $(BPFTOOL_SRC) bootstrap \
		OUTPUT=$(BPFTOOL_OUTPUT)/

# Build BPF code
$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT)
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
		-c $(filter %.c,$^) -o $@
	$(LLVM_STRIP) -g $@ # strip useless DWARF info

# Generate BPF skeletons
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL_EXEC)
	$(BPFTOOL_EXEC) gen skeleton $< > $@

# Build user-space code
$(OUTPUT)/uprobe_queue.o: uprobe_queue.c $(OUTPUT)/uprobe_queue.skel.h $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

# Build user-space code
$(OUTPUT)/uprobe_stack.o: uprobe_stack.c $(OUTPUT)/uprobe_stack.skel.h $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

# Build target.o
$(OUTPUT)/target.o: target.c | $(OUTPUT)
	$(CC) $(CFLAGS) -c $< -o $@

# Build application binary
uprobe_queue: $(OUTPUT)/uprobe_queue.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

uprobe_stack: $(OUTPUT)/uprobe_stack.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

target: $(OUTPUT)/target.o | $(OUTPUT)
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -o $@

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY: 