ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: all gemm counter clean clean-gemm clean-counter

all: gemm counter

## Build only the CUTLASS GEMM workload
gemm:
		$(MAKE) -C $(ROOT_DIR)/gemm

## Build only the launch counter probe
counter:
		$(MAKE) -C $(ROOT_DIR)/counter

clean: clean-gemm clean-counter

clean-gemm:
		$(MAKE) -C $(ROOT_DIR)/gemm clean

clean-counter:
		$(MAKE) -C $(ROOT_DIR)/counter clean
