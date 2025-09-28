# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
# bpftime LPM Trie demo Makefile

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
CFLAGS := -g -Wall -O2
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

# Application list
APPS = file_access_monitor file_access_target

# Get Clang's default includes on this system
CLANG_BPF_SYS_INCLUDES = $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-I\1|p }')

# Check if vmlinux.h exists
ifeq ($(wildcard $(VMLINUX)),)
$(error ERROR: missing $(VMLINUX))
endif

.PHONY: all clean bpf monitor target help run-demo check-env debug-bpf

all: $(APPS) bpf

bpf: $(OUTPUT)/file_access_filter.bpf.o

monitor: file_access_monitor

target: file_access_target

clean:
	@echo "  CLEAN"
	$(RM) -r $(OUTPUT) $(APPS)

# Create output directories
$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	mkdir -p $@

# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	@echo "  LIBBPF   $@"
	$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1		      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@)		      \
		    INCLUDEDIR= LIBDIR= UAPIDIR=			      \
		    install

# Build bpftool
$(BPFTOOL_EXEC): | $(BPFTOOL_OUTPUT)
	@echo "  BPFTOOL  $@"
	$(MAKE) -C $(BPFTOOL_SRC) bootstrap \
		OUTPUT=$(BPFTOOL_OUTPUT)/

# Compile BPF program
$(OUTPUT)/file_access_filter.bpf.o: file_access_filter.bpf.c $(LIBBPF_OBJ) $(VMLINUX) | $(OUTPUT)
	@echo "  BPF      $@"
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) -mllvm -bpf-stack-size=8192 \
		$(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
		-c $(filter %.c,$^) -o $@
	$(LLVM_STRIP) -g $@ # strip useless DWARF info

# Generate BPF skeleton
$(OUTPUT)/file_access_filter.skel.h: $(OUTPUT)/file_access_filter.bpf.o | $(OUTPUT) $(BPFTOOL_EXEC)
	@echo "  SKEL     $@"
	$(BPFTOOL_EXEC) gen skeleton $< > $@

# Compile userspace program object files
$(OUTPUT)/file_access_monitor.o: file_access_monitor.c $(OUTPUT)/file_access_filter.skel.h $(LIBBPF_OBJ) | $(OUTPUT)
	@echo "  CC       $@"
	$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

$(OUTPUT)/file_access_target.o: file_access_target.c | $(OUTPUT)
	@echo "  CC       $@"
	$(CC) $(CFLAGS) -c $< -o $@

# Build final executables
file_access_monitor: $(OUTPUT)/file_access_monitor.o $(LIBBPF_OBJ) | $(OUTPUT)
	@echo "  LINK     $@"
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

file_access_target: $(OUTPUT)/file_access_target.o | $(OUTPUT)
	@echo "  LINK     $@"
	$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -o $@

# Run demo
run-demo: file_access_monitor file_access_target
	@echo "=== Starting bpftime LPM Trie demo ==="
	@echo "Please run the demo script:"
	@echo "  ./run_lpm_trie_demo.sh"
	@echo ""

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Build all programs"
	@echo "  bpf         - Build BPF program"
	@echo "  monitor     - Build file access monitor program"
	@echo "  target      - Build test target program"
	@echo "  run-demo    - Show how to run complete demo"
	@echo "  clean       - Clean build files"
	@echo "  check-env   - Check build environment"
	@echo "  debug-bpf   - Debug BPF program"

# Check environment
check-env:
	@echo "=== Checking build environment ==="
	@echo -n "clang: "; which clang || echo "❌ Not found"
	@echo -n "llvm-strip: "; which llvm-strip || echo "❌ Not found"
	@echo -n "bpftool: "; which bpftool || echo "❌ Not found"
	@echo -n "System BTF: "; [ -r /sys/kernel/btf/vmlinux ] && echo "✅ Available" || echo "❌ Not available"
	@echo "BPF program size: $(shell ls -lh $(OUTPUT)/file_access_filter.bpf.o 2>/dev/null | awk '{print $$5}' || echo 'Not compiled')"

# Debug BPF program
debug-bpf: $(OUTPUT)/file_access_filter.bpf.o
	@echo "=== BPF program debug info ==="
	@echo "File size: $(shell ls -lh $(OUTPUT)/file_access_filter.bpf.o | awk '{print $$5}')"
	@echo "File type: $(shell file $(OUTPUT)/file_access_filter.bpf.o)"
	@echo "BTF info:"
	@$(BPFTOOL_EXEC) btf dump file $(OUTPUT)/file_access_filter.bpf.o 2>/dev/null || echo "  No BTF info"
	@echo "Program info:"
	@$(BPFTOOL_EXEC) prog show file $(OUTPUT)/file_access_filter.bpf.o 2>/dev/null || echo "  Cannot show program info"

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY: 