CLANG ?= clang++
LLC ?= llc
NVCC ?= nvcc
CUDA_LIBS ?= -L/usr/local/cuda/lib64/ -lcudart -pthread
PTX_TARGET_ARCH ?= sm_75
CLANG_FLAGS ?= -U__FLOAT128__ -U__SIZEOF_FLOAT128__

.PHONY: all clean

all: ../trampoline_ptx.h test

../trampoline_ptx.h: default_trampoline.ptx
	@echo "Generating trampoline_ptx.h..."
	@echo 'static const char TRAMPOLINE_PTX[] = R"(' > $@
	@grep -vE '^[[:space:]]*\\.file|^[[:space:]]*\\.loc' $< >> $@
	@echo ')";' >> $@
	@echo "Generated $@"

default_trampoline.ptx: default_trampoline.cu
	$(NVCC) -ptx -O0 -rdc=true -arch=$(PTX_TARGET_ARCH) $< -o $@

# compile it to exec for test
test: default_trampoline.cu
	$(NVCC) -lineinfo -O2 -arch=$(PTX_TARGET_ARCH) default_trampoline.cu -o test

clean:
	rm -f default_trampoline.ptx
	rm -f default_trampoline.ll
	rm -f default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s
	rm -f ../trampoline_ptx.h 
	rm -f test
