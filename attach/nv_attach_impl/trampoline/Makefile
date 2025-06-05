CLANG ?= clang++-17
CUDA_LIBS ?= -L/usr/local/cuda/lib64/ -lcudart
PTX_TARGET_ARCH ?= sm_60

.PHONY: all clean

all: ../trampoline_ptx.h test

../trampoline_ptx.h: default_trampoline-cuda-nvptx64-nvidia-cuda-sm_60.s
	@echo "Generating trampoline_ptx.h..."
	@echo 'static const char TRAMPOLINE_PTX[] = R"(' > $@
	@grep -B 10000 "// .globl	bpf_main" $< | grep -v "// .globl	bpf_main" >> $@
	@echo ')";' >> $@
	@echo "Generated $@"

default_trampoline-cuda-nvptx64-nvidia-cuda-sm_60.s: default_trampoline.cu
	$(CLANG) -S $< -Wall --cuda-gpu-arch=$(PTX_TARGET_ARCH) -O2 $(CUDA_LIBS)

# compile it to exec for test
test: default_trampoline-cuda-nvptx64-nvidia-cuda-sm_60.s
	$(CLANG) -o test --cuda-gpu-arch=$(PTX_TARGET_ARCH) default_trampoline.cu $(CUDA_LIBS)

clean:
	rm -f default_trampoline-cuda-nvptx64-nvidia-cuda-sm_60.s
	rm -f ../trampoline_ptx.h 