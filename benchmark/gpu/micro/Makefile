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
# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated
INCLUDES := -I$(OUTPUT) -I../../../third_party/libbpf/include/uapi -I$(dir $(VMLINUX))
CFLAGS := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

APPS = cuda_probe cuda_probe_kernel_gpu# minimal minimal_legacy uprobe kprobe fentry usdt sockfilter tc ksyscall

CARGO ?= $(shell which cargo)
ifeq ($(strip $(CARGO)),)
BZS_APPS :=
else
BZS_APPS := # profile
APPS += $(BZS_APPS)
# Required by libblazesym
ALL_LDFLAGS += -lrt -ldl -lpthread -lm
endif

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architecture-specific dirs will be "missing" on some architectures/distros -
# headers such as asm/types.h, asm/byteorder.h, asm/socket.h, asm/sockios.h,
# sys/cdefs.h etc. might be missing.
#
# Use '-idirafter': Don't interfere with include mechanics except where the
# build would have failed anyways.
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
# NVBit tool compilation
NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

ifeq ($(DEBUG),1)
NVCC_OPT=-O0 -g
else
NVCC_OPT=-O3
endif

# Path to NVBit - NVBit is NOT part of CUDA Toolkit, will be auto-downloaded if needed
# Adjust this to your system's NVBit installation location
NVBIT_PATH?=$(HOME)/nvbit_release_x86_64/core
NVBIT_RELEASE_DIR?=$(HOME)/nvbit_release_x86_64
NVBIT_INCLUDES=-I$(NVBIT_PATH)

NVBIT_LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

NVBIT_SOURCES=nvbit_vec_add.cu nvbit_timing_funcs.cu
NVBIT_OBJECTS=$(NVBIT_SOURCES:.cu=.o)
CUDA_ARCH?=all

# Auto-download and install NVBit if not found
$(NVBIT_PATH)/libnvbit.a:
	@echo "NVBit not found, downloading version 1.7.6 with SM_120 support..."
	@mkdir -p $(HOME)
	@cd $(HOME) && \
	rm -rf nvbit_release_x86_64 && \
	wget -q https://github.com/NVlabs/NVBit/releases/download/v1.7.6/nvbit-Linux-x86_64-1.7.6.tar.bz2 && \
	tar xjf nvbit-Linux-x86_64-1.7.6.tar.bz2 && \
	rm nvbit-Linux-x86_64-1.7.6.tar.bz2
	@echo "NVBit 1.7.6 installed to $(NVBIT_RELEASE_DIR)"

nvbit_vec_add.so: nvbit_vec_add.o nvbit_timing_funcs.o $(NVBIT_PATH)/libnvbit.a
	g++ -shared -fPIC nvbit_vec_add.o nvbit_timing_funcs.o \
		$(NVBIT_LIBS) $(NVCC_PATH) -lcuda -lcudart_static -lpthread -ldl -o $@

nvbit_vec_add.o: nvbit_vec_add.cu
	$(NVCC) -c -std=c++11 $(NVBIT_INCLUDES) -Xcompiler -Wall $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

nvbit_timing_funcs.o: nvbit_timing_funcs.cu
	$(NVCC) $(NVBIT_INCLUDES) -Xptxas -astoolspatch --keep-device-functions -arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

.PHONY: clean
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf $(OUTPUT) $(APPS) nvbit_vec_add.so *.o

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


$(LIBBLAZESYM_SRC)/target/release/libblazesym.a::
	$(Q)cd $(LIBBLAZESYM_SRC) && $(CARGO) build --features=cheader,dont-generate-test-files --release

$(LIBBLAZESYM_OBJ): $(LIBBLAZESYM_SRC)/target/release/libblazesym.a | $(OUTPUT)
	$(call msg,LIB, $@)
	$(Q)cp $(LIBBLAZESYM_SRC)/target/release/libblazesym.a $@

$(LIBBLAZESYM_HEADER): $(LIBBLAZESYM_SRC)/target/release/libblazesym.a | $(OUTPUT)
	$(call msg,LIB,$@)
	$(Q)cp $(LIBBLAZESYM_SRC)/target/release/blazesym.h $@

# Build BPF code
$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	$(Q)$(CLANG) -Xlinker --export-dynamic -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)		      \
		     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES)		      \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Generate BPF skeletons
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

# Build user-space code
$(patsubst %,$(OUTPUT)/%.o,$(APPS)): %.o: %.skel.h

$(OUTPUT)/%.o: %.c $(wildcard %.h) | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $(filter %.c,$^) -o $@

$(patsubst %,$(OUTPUT)/%.o,$(BZS_APPS)): $(LIBBLAZESYM_HEADER)

$(BZS_APPS): $(LIBBLAZESYM_OBJ)

# Build application binary
$(APPS): %: $(OUTPUT)/%.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -o $@

# Run the CUDA vector addition benchmark with NVBit instrumentation
run_nvbit: nvbit_vec_add.so
	CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so ./vec_add

# Run with verbose output
run_nvbit_verbose: nvbit_vec_add.so
	CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so TOOL_VERBOSE=1 ./vec_add

nvbit: nvbit_vec_add.so

# delete failed targets
.DELETE_ON_ERROR:

# keep intermediate (.skel.h, .bpf.o, etc) targets
.SECONDARY:
