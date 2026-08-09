# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)

NVCC ?= nvcc
NVCC_FLAGS = -cudart shared -g -Wno-deprecated-gpu-targets
INCLUDE_FLAGS = -I. -Isample_headers
CXX_FLAGS = -std=c++11

.PHONY: all
all: vec_add matrixMul simpleCUBLAS

vec_add: vec_add.cu
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) $(NVCC_FLAGS) -O3 vec_add.cu -g -o vec_add; \
	else \
		echo "Warning: CUDA not found, skipping vec_add build"; \
	fi

matrixMul: matrixMul.cu
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) $(NVCC_FLAGS) -O3 $(INCLUDE_FLAGS) $(CXX_FLAGS) matrixMul.cu -o matrixMul; \
	else \
		echo "Warning: CUDA not found, skipping matrixMul build"; \
	fi

simpleCUBLAS: simpleCUBLAS.cpp
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CXX_FLAGS) simpleCUBLAS.cpp -o simpleCUBLAS -lcublas; \
	else \
		echo "Warning: CUDA not found, skipping simpleCUBLAS build"; \
	fi

.PHONY: clean
clean:
	rm -f vec_add matrixMul simpleCUBLAS
