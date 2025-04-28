CXX = g++
CC = gcc
CXXFLAGS = -Wall -Wextra -fpic -O2 -std=c++17
CFLAGS = -Wall -Wextra -fpic -O2
LDFLAGS = -shared -fPIC

# RLBox repository path (can be overridden)
RLBOX_PATH ?= $(shell pwd)/rlbox
RLBOX_WASM2C_PATH ?= $(shell pwd)/rlbox_wasm2c_sandbox
WASI_SDK_PATH ?= /opt/wasi-sdk

# Include paths - make sure they cover all possible locations
INCLUDES = -I$(RLBOX_PATH) \
           -I$(RLBOX_WASM2C_PATH) \
           -I$(RLBOX_PATH)/code/include \
           -I$(RLBOX_WASM2C_PATH)/include \
           -I/usr/local/include \
           -I/usr/include

.PHONY: all clean install-deps install-rlbox install-wasi-sdk noop wasm2c

all: noop

# Build the noop sandbox version (for easier development and testing)
noop: libfilter_rlbox.so

# Build the wasm2c sandbox version (for real isolation)
wasm2c: mylib.wasm libfilter_rlbox_wasm2c.so

# Clone and install RLBox
install-rlbox:
	@echo "Installing RLBox framework..."
	@if [ ! -d "$(RLBOX_PATH)" ]; then \
		git clone https://github.com/PLSysSec/rlbox $(RLBOX_PATH); \
	else \
		echo "RLBox directory already exists. Skipping clone."; \
	fi
	
	@echo "Installing RLBox wasm2c sandbox..."
	@if [ ! -d "$(RLBOX_WASM2C_PATH)" ]; then \
		git clone https://github.com/PLSysSec/rlbox_wasm2c_sandbox $(RLBOX_WASM2C_PATH); \
	else \
		echo "RLBox wasm2c sandbox directory already exists. Skipping clone."; \
	fi
	
	@echo "Building RLBox wasm2c sandbox..."
	@if [ ! -d "$(RLBOX_WASM2C_PATH)/build" ]; then \
		mkdir -p $(RLBOX_WASM2C_PATH)/build; \
	fi
	cd $(RLBOX_WASM2C_PATH)/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4

	@echo "Building RLBox..."
	@if [ ! -d "$(RLBOX_PATH)/build" ]; then \
		mkdir -p $(RLBOX_PATH)/build; \
	fi
	cd $(RLBOX_PATH) && cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release && cmake --build ./build --config Release --parallel
	
	@echo "RLBox installation complete!"
	@echo "Adding symbolic links..."
	@if [ ! -d "include/rlbox" ]; then \
		mkdir -p include/rlbox; \
		ln -sf $(RLBOX_PATH)/code/include/*.hpp include/rlbox/; \
		ln -sf $(RLBOX_WASM2C_PATH)/include/*.hpp include/rlbox/; \
	fi

# Install WASI SDK (for WASM compilation)
install-wasi-sdk:
	@echo "Downloading and installing WASI SDK to $(WASI_SDK_PATH)..."
	@if [ ! -d "$(WASI_SDK_PATH)" ]; then \
		wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-19/wasi-sdk-19.0-linux.tar.gz && \
		tar xf wasi-sdk-19.0-linux.tar.gz && \
		sudo mv wasi-sdk-19.0 $(WASI_SDK_PATH) && \
		rm wasi-sdk-19.0-linux.tar.gz; \
	else \
		echo "WASI SDK directory already exists. Skipping installation."; \
	fi

# Install all dependencies
install-deps: install-rlbox install-wasi-sdk

# Compile the sandboxed library
mylib.o: mylib.c mylib.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile the RLBox wrapper with noop sandbox
rlbox_filter_noop.o: rlbox_filter.cpp rlbox_filter.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Create shared library with noop sandbox
libfilter_rlbox.so: mylib.o rlbox_filter_noop.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile to WASM for wasm2c sandbox
mylib.wasm: mylib.c mylib.h
	@if [ ! -x "$(WASI_SDK_PATH)/bin/clang" ]; then \
		echo "WASI SDK not found at $(WASI_SDK_PATH). Please run 'make install-wasi-sdk' first."; \
		exit 1; \
	fi
	$(WASI_SDK_PATH)/bin/clang \
		-O2 \
		-nostdlib \
		--target=wasm32-wasi \
		--sysroot=$(WASI_SDK_PATH)/share/wasi-sysroot \
		-Wl,--export=initialize \
		-Wl,--export=url_filter \
		-Wl,--export=get_counters \
		-Wl,--export=memory \
		-Wl,--no-entry \
		-Wl,--allow-undefined \
		-o $@ $<

# Compile the RLBox wrapper with wasm2c sandbox
rlbox_filter_wasm2c.o: rlbox_filter_wasm2c.cpp rlbox_filter.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Create shared library with wasm2c sandbox
libfilter_rlbox_wasm2c.so: rlbox_filter_wasm2c.o
	$(CXX) $(LDFLAGS) -o $@ $^ -L$(RLBOX_WASM2C_PATH)/build -lwasm2c

clean:
	rm -f *.o *.so *.wasm *.wasm.c *.wasm.h 