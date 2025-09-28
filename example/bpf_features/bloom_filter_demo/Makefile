# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
OUTPUT := .output
CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool
LIBBPF_SRC := $(abspath ../../../third_party/libbpf/src)
BPFTOOL_SRC := $(abspath ../../../third_party/bpftool/src)
# Build libbpf locally into .output
LIBBPF_OBJ := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(OUTPUT)/bpftool)
# Try to find bpftool from multiple locations, or build our own
BPFTOOL_BIN := $(shell \
	if [ -f "../../../build/bpftool/src/bpftool" ]; then \
		echo "../../../build/bpftool/src/bpftool"; \
	elif [ -f "../../../third_party/bpftool/src/bpftool" ]; then \
		echo "../../../third_party/bpftool/src/bpftool"; \
	elif [ -f "$(BPFTOOL_OUTPUT)/bootstrap/bpftool" ]; then \
		echo "$(BPFTOOL_OUTPUT)/bootstrap/bpftool"; \
	else \
		echo "$(BPFTOOL_OUTPUT)/bootstrap/bpftool"; \
	fi)
ARCH := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/' | sed 's/ppc64le/powerpc/' | sed 's/mips.*/mips/')
VMLINUX := ../../../third_party/vmlinux/$(ARCH)/vmlinux.h
# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated

# Headers search: local output and vendored headers
INCLUDES := -I$(OUTPUT) -I../../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX)) \
	-I$(BPFTOOL_OUTPUT)/bootstrap/libbpf/include \
	-I../../../third_party/bpftool/build/usr/include \
	-I../../../third_party/libbpf/build/usr/include
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

APPS = uprobe_bloom_filter target

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architectures/distros don't find standard headers.
CLANG_BPF_SYS_INCLUDES = $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \* | |; /^ /p }')

ifeq ($(V),1)
	Q =
	msg =
else
	Q = @
	msg = @printf '  %-8s %s%s\n'					\
		      "$(1)"						\
		      "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))"	\
		      "$(if $(3), $(3))";
	MAKEFLAGS += --no-print-directory
endif

.PHONY: all
all: $(APPS)

.PHONY: clean
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf $(OUTPUT) $(APPS)

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

# Build libbpf locally if not present
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1 \
	    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@) \
	    INCLUDEDIR= LIBDIR= UAPIDIR= \
	    install

# Build bpftool if it doesn't exist
$(BPFTOOL_BIN): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)if [ ! -f "$(BPFTOOL_BIN)" ]; then \
		$(MAKE) -C $(BPFTOOL_SRC) bootstrap OUTPUT=$(BPFTOOL_OUTPUT)/ || \
		$(MAKE) -C ../../../third_party/bpftool/src bootstrap OUTPUT=$(BPFTOOL_OUTPUT)/; \
	fi

# Build BPF code (ensure headers are installed via local libbpf)
$(OUTPUT)/%.bpf.o: %.bpf.c $(wildcard %.h) $(VMLINUX) $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BPF,$@)
	$(Q)$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) -c $(filter %.c,$^) -o $@
	$(Q)$(LLVM_STRIP) -g $@ # strip useless DWARF info

# Generate BPF skeletons
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o $(BPFTOOL_BIN) | $(OUTPUT)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL_BIN) gen skeleton $< > $@

# Build user-space code - only uprobe_bloom_filter needs skeleton
$(OUTPUT)/uprobe_bloom_filter.o: uprobe_bloom_filter.c $(OUTPUT)/uprobe_bloom_filter.skel.h $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

$(OUTPUT)/target.o: target.c | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

# Build application binary
uprobe_bloom_filter: $(OUTPUT)/uprobe_bloom_filter.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

target: $(OUTPUT)/target.o | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -o $@

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY: 