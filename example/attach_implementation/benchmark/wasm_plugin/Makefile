CC = gcc
CFLAGS = -Wall -Wextra -fpic -O2 -g
EMCC = emcc
EMCC_FLAGS = -O2 -s WASM=1 -s STANDALONE_WASM=1 -s EXPORTED_FUNCTIONS="['_initialize', '_url_filter', '_get_counters']" -s -s ALLOW_MEMORY_GROWTH=1 -s ERROR_ON_UNDEFINED_SYMBOLS=0
LDFLAGS = -shared

# Check if WASM3_DIR is defined, otherwise use default path
ifndef WASM3_DIR
WASM3_DIR = $(PWD)/wasm3/source
endif

# Default target - build the WebAssembly runtime and module
all: wasm3_lib url_filter.wasm libwasm_filter.so

# Create the WebAssembly module if Emscripten is found
url_filter.wasm: url_filter.c
	@if command -v $(EMCC) > /dev/null 2>&1; then \
		echo "Building WebAssembly module with Emscripten..."; \
		$(EMCC) $(EMCC_FLAGS) -o $@ $<; \
	else \
		echo "Emscripten not found. Please install Emscripten or provide a pre-built url_filter.wasm file."; \
		echo "You can install Emscripten with: make install-deps"; \
		exit 1; \
	fi

# WASM runtime wrapper
wasm_runtime.o: wasm_runtime.c wasm_runtime.h wasm3_lib
	$(CC) $(CFLAGS) -I$(WASM3_DIR) -c -o $@ $<

# Shared library that implements the filter interface
libwasm_filter.so: wasm_runtime.o
	$(CC) $(LDFLAGS) -o $@ $< -L$(PWD)/wasm3/build/source -lm3

# Build wasm3 locally with -fPIC
wasm3_lib:
	@if [ ! -d "wasm3" ]; then \
		echo "Cloning wasm3 repository..."; \
		git clone https://github.com/wasm3/wasm3.git; \
	fi
	@if [ ! -f "wasm3/build/source/libm3.a" ]; then \
		echo "Building wasm3 library with -fPIC..."; \
		mkdir -p wasm3/build && cd wasm3/build && \
		cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS="-fPIC" .. && make; \
	fi

# Clean up
clean:
	rm -f *.o *.so *.wasm
	rm -rf wasm3

# Full install of dependencies (for Ubuntu/Debian)
install-deps:
	@echo "Installing build dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake git python3
	
	@echo "Installing Emscripten..."
	git clone https://github.com/emscripten-core/emsdk.git
	cd emsdk && ./emsdk install latest && ./emsdk activate latest
	@echo "Please run 'source ./emsdk/emsdk_env.sh' to set up Emscripten environment"

.PHONY: all clean install-deps wasm3_lib 