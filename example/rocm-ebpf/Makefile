# SPDX-License-Identifier: MIT
OUTPUT := $(abspath .output)
BPFTIME_ROOT := $(abspath ../..)
CLANG ?= clang
CXXFLAGS ?= -O2 -g -Wall -Wextra
ROCM_PATH ?= $(shell hipconfig --rocmpath)
HIPCXX ?= $(ROCM_PATH)/lib/llvm/bin/clang++
GPU_ARCH ?= native
HIPFLAGS ?= -O2 -g
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/;s/aarch64/arm64/;s/ppc64le/powerpc/;s/riscv64/riscv/;s/loongarch64/loongarch/')
VMLINUX := $(BPFTIME_ROOT)/third_party/vmlinux/$(ARCH)/vmlinux.h
LIBBPF_SRC := $(BPFTIME_ROOT)/third_party/libbpf/src
LIBBPF_OBJ := $(OUTPUT)/libbpf.a
HELPER_INCLUDE := $(BPFTIME_ROOT)/attach/rocm_attach_impl
APPS := rocm_ebpf victim

.PHONY: all clean
.DELETE_ON_ERROR:
all: $(APPS)

$(OUTPUT)/libbpf:
	mkdir -p "$@"

# Build a private libbpf copy, like the other standalone examples.
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(OUTPUT)/libbpf
	$(MAKE) -C "$(LIBBPF_SRC)" BUILD_STATIC_ONLY=1 \
		OBJDIR="$(OUTPUT)/libbpf" DESTDIR="$(OUTPUT)/" \
		INCLUDEDIR= LIBDIR= UAPIDIR= install

$(OUTPUT)/rocm_ebpf.bpf.o: rocm_ebpf.bpf.c $(HELPER_INCLUDE)/bpf_test_println.h $(VMLINUX) $(LIBBPF_OBJ) Makefile
	$(CLANG) -target bpf -O2 -g -D__TARGET_ARCH_$(ARCH) \
		-I"$(dir $(VMLINUX))" -I"$(OUTPUT)" -I"$(HELPER_INCLUDE)" \
		-c "$<" -o "$@"

rocm_ebpf: rocm_ebpf.cpp $(OUTPUT)/rocm_ebpf.bpf.o $(LIBBPF_OBJ) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -I"$(OUTPUT)" \
		-DBPFTIME_EXAMPLE_BPF='"$(OUTPUT)/rocm_ebpf.bpf.o"' \
		"$<" "$(LIBBPF_OBJ)" $(LDFLAGS) -lelf -lz $(LDLIBS) -o "$@"

victim: victim.hip Makefile
	"$(HIPCXX)" -x hip -std=c++17 $(HIPFLAGS) \
		--offload-arch="$(GPU_ARCH)" --rocm-path="$(ROCM_PATH)" --hip-path="$(ROCM_PATH)" \
		"$<" -L"$(ROCM_PATH)/lib" -Wl,-rpath,"$(ROCM_PATH)/lib" -lamdhip64 -o "$@"

clean:
	rm -rf -- "$(OUTPUT)" $(APPS)
