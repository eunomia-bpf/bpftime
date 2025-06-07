CLANG ?= nvcc
CUDA_LIBS ?= -L/usr/local/cuda/lib64/ -lcudart

# Create a temporary file to detect SM version
DETECT_SM := $(shell echo '#include <cuda_runtime.h>\n#include <stdio.h>\nint main() { cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); printf("sm_%d%d", prop.major, prop.minor); return 0; }' > detect_sm.cu && $(CLANG) -o detect_sm detect_sm.cu $(CUDA_LIBS) && ./detect_sm && rm detect_sm detect_sm.cu)

# Use detected SM version or fallback to sm_90
PTX_TARGET_ARCH ?= $(if $(DETECT_SM),$(DETECT_SM),sm_90)

.PHONY: all clean ../trampoline_ptx.h

all: ../trampoline_ptx.h test

../trampoline_ptx.h: default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s
	@echo "Generating trampoline_ptx.h..."
	@echo 'static const char TRAMPOLINE_PTX[] = R"(' > $@
	@grep -B 10000 "__device_stub__bpf_main" $< | grep -v "__device_stub__bpf_main" >> $@
	@echo ')";' >> $@
	@echo "Generated $@"

default_trampoline-cuda-nvptx64-nvidia-cuda-$(PTX_TARGET_ARCH).s: default_trampoline.cu
	@echo "Using CUDA architecture: $(PTX_TARGET_ARCH)"
	$(CLANG) -S $< -Wall -O2 -arch=$(PTX_TARGET_ARCH) -o $@

# compile it to exec for test
test: default_trampoline.cu
	@echo "Using CUDA architecture: $(PTX_TARGET_ARCH)"
	$(CLANG) -o test -arch=$(PTX_TARGET_ARCH) $< $(CUDA_LIBS)

clean:
	rm -f default_trampoline-cuda-nvptx64-nvidia-cuda-*.s
	rm -f ../trampoline_ptx.h 