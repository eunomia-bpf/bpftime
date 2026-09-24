# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
# GPU-local full-record device buffer (32-bank GPU_ARRAY map, type 1503).
# Build from a directory three levels below the bpftime root, i.e. the
# example/gpu/* tree layout, so the ../../../third_party paths resolve.
# LAYOUT selects the bank value layout (opt-in, default off):
#   aos: the measured record-major AoS layout; builds the .output/ object
#        tree and the full-record-device-buffer binary (default semantics
#        unchanged).
#   soa: the field-major SoA layout; builds a separate .output-soa/ object
#        tree and the distinct full-record-device-buffer-soa binary, so
#        toggling LAYOUT can never reuse objects from the other layout.
#   aosoa: the 32-slot grouped SoA (AoSoA) layout; builds a separate
#        .output-aosoa/ object tree and the distinct
#        full-record-device-buffer-aosoa binary, so toggling LAYOUT can
#        never reuse objects from the other layouts.
#   soa-warp: the field-major SoA layout with the opt-in warp-contiguous
#        storage index (FRDB_SOA_LAYOUT + FRDB_WARP_CONTIGUOUS_INDEX);
#        the bank value is byte-identical to soa, only the writer's slot
#        bijection changes. Builds a separate .output-soa-warp/ object
#        tree and the distinct full-record-device-buffer-soa-warp binary.
LAYOUT ?= aos
ifneq ($(LAYOUT),aos)
LAYOUT_SUFFIX := -$(LAYOUT)
endif
ifeq ($(LAYOUT),soa)
EXTRA_LAYOUT_CFLAGS := -DFRDB_SOA_LAYOUT
else ifeq ($(LAYOUT),aosoa)
EXTRA_LAYOUT_CFLAGS := -DFRDB_AOSOA_LAYOUT
else ifeq ($(LAYOUT),soa-warp)
EXTRA_LAYOUT_CFLAGS := -DFRDB_SOA_LAYOUT -DFRDB_WARP_CONTIGUOUS_INDEX
else ifneq ($(LAYOUT),aos)
$(error unsupported LAYOUT '$(LAYOUT)'; use aos, soa, aosoa, or soa-warp)
endif
OUTPUT := .output$(LAYOUT_SUFFIX)
CLANG ?= clang
CUDA_HOME ?= /usr/local/cuda
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
INCLUDES := -I$(OUTPUT) -I../../../runtime/include -I$(CUDA_HOME)/include \
	    -I../../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX))
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS) -lcuda

APPS = full-record-device-buffer$(LAYOUT_SUFFIX)
APP_SRC = full-record-device-buffer

CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

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

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

$(call allow-override,CC,$(CROSS_COMPILE)cc)
$(call allow-override,LD,$(CROSS_COMPILE)ld)

.PHONY: all
all: $(APPS)

.PHONY: clean
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf .output .output-soa .output-aosoa .output-soa-warp \
	    full-record-device-buffer full-record-device-buffer-soa \
	    full-record-device-buffer-aosoa full-record-device-buffer-soa-warp

$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1		      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@)		      \
		    INCLUDEDIR= LIBDIR= UAPIDIR=			      \
		    install

# Build bpftool
$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)$(MAKE) ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap

# Build BPF code
$(OUTPUT)/$(APP_SRC).bpf.o: $(APP_SRC).bpf.c $(LIBBPF_OBJ) $(wildcard *.h) $(VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	$(Q)$(CLANG) -Xlinker --export-dynamic -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		     $(INCLUDES) $(EXTRA_LAYOUT_CFLAGS) $(CLANG_BPF_SYS_INCLUDES) \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Generate BPF skeletons
$(OUTPUT)/$(APP_SRC).skel.h: $(OUTPUT)/$(APP_SRC).bpf.o | $(OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

# Build user-space code
$(OUTPUT)/$(APP_SRC).o: $(APP_SRC).c $(wildcard *.h) $(OUTPUT)/$(APP_SRC).skel.h | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(EXTRA_LAYOUT_CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

# Build application binary
$(APPS): $(OUTPUT)/$(APP_SRC).o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY:
