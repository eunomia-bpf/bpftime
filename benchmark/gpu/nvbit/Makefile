

# NVBit tool compilation
NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

ifeq ($(DEBUG),1)
NVCC_OPT=-O0 -g
else
NVCC_OPT=-O3
endif

all: nvbit_vec_add.so

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

nvbit_vec_add.so: vec_add nvbit_vec_add.o nvbit_timing_funcs.o $(NVBIT_PATH)/libnvbit.a
	g++ -shared -fPIC nvbit_vec_add.o nvbit_timing_funcs.o \
		$(NVBIT_LIBS) $(NVCC_PATH) -lcuda -lcudart_static -lpthread -ldl -o $@

nvbit_vec_add.o: nvbit_vec_add.cu
	$(NVCC) -c -std=c++11 $(NVBIT_INCLUDES) -Xcompiler -Wall $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

nvbit_timing_funcs.o: nvbit_timing_funcs.cu
	$(NVCC) $(NVBIT_INCLUDES) -Xptxas -astoolspatch --keep-device-functions -arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

.PHONY: clean
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf $(OUTPUT) $(APPS) vec_add nvbit_vec_add.so *.o


# Run the CUDA vector addition benchmark with NVBit instrumentation
run_nvbit: nvbit_vec_add.so
	CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so ./vec_add

# Run with verbose output
run_nvbit_verbose: nvbit_vec_add.so
	CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so TOOL_VERBOSE=1 ./vec_add

nvbit: nvbit_vec_add.so


vec_add: vec_add.cu
	@if command -v nvcc >/dev/null 2>&1; then \
		nvcc -cudart shared vec_add.cu -o vec_add -g; \
	else \
		echo "Warning: CUDA not found, skipping vec_add build"; \
	fi
