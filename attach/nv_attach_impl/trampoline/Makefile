CLANG ?= clang++-17
CUDA_LIBS ?= -L/usr/local/cuda/lib64/ -lcudart -pthread
PTX_TARGET_ARCH ?= sm_61
CLANG_FLAGS ?= -U__FLOAT128__ -U__SIZEOF_FLOAT128__
PYTHON ?= python3

.PHONY: all clean test-ringbuf-protocol

all: ../trampoline_ptx.h test test-ringbuf-protocol

test-ringbuf-protocol: default_trampoline.cu ../trampoline_ptx.h \
                       test_ringbuf_protocol.py \
                       ../../../runtime/include/bpf_attach_ctx.hpp
	$(PYTHON) test_ringbuf_protocol.py default_trampoline.cu \
		../trampoline_ptx.h ../../../runtime/include/bpf_attach_ctx.hpp

../trampoline_ptx.h: default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s
	@echo "Generating trampoline_ptx.h..."
	@echo 'static const char TRAMPOLINE_PTX[] = R"(' > $@
	@grep -B 10000 "// .globl	bpf_main" $< | grep -v "// .globl	bpf_main" >> $@
	@echo ')";' >> $@
	@echo "Generated $@"

default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s: default_trampoline.cu
	$(CLANG) -S $< -Wall --cuda-gpu-arch=$(PTX_TARGET_ARCH) -O2 $(CLANG_FLAGS) $(CUDA_LIBS)

# compile it to exec for test
test: default_trampoline.cu
	$(CLANG) -o test --cuda-gpu-arch=$(PTX_TARGET_ARCH) default_trampoline.cu $(CLANG_FLAGS) $(CUDA_LIBS)

clean:
	rm -f default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s
	rm -f ../trampoline_ptx.h 
