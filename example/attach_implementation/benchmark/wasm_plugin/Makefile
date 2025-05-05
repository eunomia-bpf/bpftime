CC = gcc
CFLAGS = -Wall -Wextra -fpic -O2 -g

# Find WASI-SDK
WASI_SDK_PATH ?= /opt/wasi-sdk
WASI_SYSROOT ?= $(WASI_SDK_PATH)/share/wasi-sysroot
WASI_CC = $(WASI_SDK_PATH)/bin/clang
WASI_AR = $(WASI_SDK_PATH)/bin/llvm-ar
WASI_FLAGS = -O2 \
             -nostdlib \
             --target=wasm32-wasi \
             --sysroot=$(WASI_SYSROOT) \
             -Wl,--export=initialize \
             -Wl,--export=url_filter \
             -Wl,--export=get_counters \
             -Wl,--export=set_buffer \
             -Wl,--export=get_buffer \
             -Wl,--export=memory \
             -Wl,--no-entry \
             -Wl,--allow-undefined \
             -Wl,--strip-all

LDFLAGS = -shared

# Check if WAMR_DIR is defined, otherwise use default path
ifndef WAMR_DIR
WAMR_DIR = $(PWD)/wasm-micro-runtime
endif

# WAMR core include paths - update these to match WAMR structure
WAMR_INCLUDE = -I$(WAMR_DIR)/core/iwasm/include \
               -I$(WAMR_DIR)/core/shared/utils \
               -I$(WAMR_DIR)/core/shared/platform/include

# Default target - build the WebAssembly runtime and module
all: wamr_lib url_filter.wasm libwasm_filter.so

# Create the WebAssembly module using WASI-SDK
url_filter.wasm: url_filter.c
	@if [ -x "$(WASI_CC)" ]; then \
		echo "Building WebAssembly module with WASI-SDK..."; \
		$(WASI_CC) $(WASI_FLAGS) -o $@ $<; \
	else \
		echo "WASI-SDK not found at $(WASI_SDK_PATH)"; \
		echo "Please install WASI-SDK or set WASI_SDK_PATH environment variable"; \
		exit 1; \
	fi

# WASM runtime wrapper
wasm_runtime.o: wasm_runtime.c wasm_runtime.h wamr_lib
	$(CC) $(CFLAGS) $(WAMR_INCLUDE) -c -o $@ $<

# Build static iwasm library instead of using shared
libiwasm.a: wamr_lib
	@if [ ! -f "$(WAMR_DIR)/product-mini/platforms/linux/build/libiwasm.a" ]; then \
		echo "Building WAMR static library..."; \
		cd $(WAMR_DIR)/product-mini/platforms/linux/build && \
		cmake -DWAMR_BUILD_INTERP=0 \
		      -DWAMR_BUILD_FAST_INTERP=0 \
		      -DWAMR_BUILD_JIT=1 \
		      -DWAMR_BUILD_LIBC_BUILTIN=1 \
		      -DWAMR_BUILD_LIBC_WASI=0 \
		      -DWAMR_BUILD_MULTI_MODULE=0 \
		      -DCMAKE_BUILD_TYPE=Release \
		      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		      -DBUILD_SHARED_LIBS=0 \
		      .. && \
		make -j$(nproc); \
	fi
	cp $(WAMR_DIR)/product-mini/platforms/linux/build/libiwasm.a ./libiwasm.a

# Shared library that implements the filter interface
libwasm_filter.so: wasm_runtime.o libiwasm.a
	$(CC) $(LDFLAGS) -o $@ $< -L. -liwasm -lm -ldl -lpthread -Wl,-Bstatic -Wl,-Bdynamic

# Build WAMR locally
wamr_lib:
	@if [ ! -d "$(WAMR_DIR)" ]; then \
		echo "Cloning WAMR repository..."; \
		git clone https://github.com/bytecodealliance/wasm-micro-runtime.git $(WAMR_DIR); \
		cd $(WAMR_DIR) && git checkout a9966897b65914362a6835c26a8bd20372ec6053; \
	fi
	@if [ ! -d "$(WAMR_DIR)/product-mini/platforms/linux/build" ]; then \
		echo "Creating WAMR build directory..."; \
		mkdir -p $(WAMR_DIR)/product-mini/platforms/linux/build; \
	fi
	@if [ ! -f "$(WAMR_DIR)/product-mini/platforms/linux/build/CMakeCache.txt" ]; then \
		echo "Configuring WAMR build..."; \
		cd $(WAMR_DIR)/product-mini/platforms/linux/build && \
		cmake -DWAMR_BUILD_INTERP=1 \
		      -DWAMR_BUILD_FAST_INTERP=1 \
		      -DWAMR_BUILD_JIT=0 \
		      -DWAMR_BUILD_LIBC_BUILTIN=1 \
		      -DWAMR_BUILD_LIBC_WASI=0 \
		      -DWAMR_BUILD_MULTI_MODULE=0 \
		      -DCMAKE_BUILD_TYPE=Release \
		      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		      -DBUILD_SHARED_LIBS=0 \
		      ..; \
	fi
	@if [ ! -f "$(WAMR_DIR)/product-mini/platforms/linux/build/libiwasm.a" ]; then \
		echo "Building WAMR library..."; \
		cd $(WAMR_DIR)/product-mini/platforms/linux/build && \
		make -j$(nproc); \
	fi

# Clean up
clean:
	rm -f *.o *.so *.wasm *.a
	rm -rf $(WAMR_DIR)

# Full install of dependencies (for Ubuntu/Debian)
install-deps:
	@echo "Installing build dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake git python3 libpthread-stubs0-dev
	
	@echo "Installing WASI-SDK..."
	wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-19/wasi-sdk-19.0-linux.tar.gz
	tar xf wasi-sdk-19.0-linux.tar.gz
	sudo mv wasi-sdk-19.0 /opt/wasi-sdk
	rm wasi-sdk-19.0-linux.tar.gz

.PHONY: all clean install-deps wamr_lib 