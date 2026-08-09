CUDA_HOME ?= /usr/local/cuda-12.6
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUTLASS_GIT_URL ?= https://github.com/NVIDIA/cutlass.git
CUTLASS_GIT_REF ?= v3.5.0
CUTLASS_DIR ?= $(abspath $(CURDIR)/.deps/cutlass)
TARGET := cutlass_gemm
SRC := cutlass_gemm.cu

NVCC_FLAGS ?= -O3 -std=c++17 --expt-relaxed-constexpr --use_fast_math
INCLUDES := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
LDLIBS := -lcuda -lcudart

.PHONY: all clean deps

all: $(TARGET)

deps: $(CUTLASS_DIR)

$(CUTLASS_DIR):
	@echo "Fetching CUTLASS into $(CUTLASS_DIR)"
	@mkdir -p $(dir $@)
	@git clone --depth 1 --branch $(CUTLASS_GIT_REF) $(CUTLASS_GIT_URL) $(CUTLASS_DIR)

$(TARGET): $(SRC) | $(CUTLASS_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)
