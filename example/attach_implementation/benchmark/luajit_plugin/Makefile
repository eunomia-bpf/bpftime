CC = gcc
CFLAGS = -Wall -Wextra -fpic -O2
LDFLAGS = -shared

# Try to detect LuaJIT with pkg-config first
LUAJIT_CFLAGS := $(shell pkg-config --cflags luajit 2>/dev/null)
LUAJIT_LDFLAGS := $(shell pkg-config --libs luajit 2>/dev/null)

# If pkg-config failed, try some common paths
ifeq ($(LUAJIT_CFLAGS)$(LUAJIT_LDFLAGS),)
  # Common paths for LuaJIT headers and libs
  ifneq ($(wildcard /usr/include/luajit-2.1),)
    LUAJIT_CFLAGS = -I/usr/include/luajit-2.1
    LUAJIT_LDFLAGS = -lluajit-5.1
  else ifneq ($(wildcard /usr/include/luajit-2.0),)
    LUAJIT_CFLAGS = -I/usr/include/luajit-2.0
    LUAJIT_LDFLAGS = -lluajit-5.1
  else ifneq ($(wildcard /usr/local/include/luajit-2.1),)
    LUAJIT_CFLAGS = -I/usr/local/include/luajit-2.1
    LUAJIT_LDFLAGS = -L/usr/local/lib -lluajit-5.1
  else
    # Try to download LuaJIT source and build locally (if nothing else works)
    $(info LuaJIT not found, will attempt to build from source)
    LUAJIT_LOCAL_DIR = luajit-local
    LUAJIT_CFLAGS = -I$(LUAJIT_LOCAL_DIR)/src
    LUAJIT_LDFLAGS = -L$(LUAJIT_LOCAL_DIR)/src -lluajit
    BUILD_LOCAL_LUAJIT = 1
  endif
endif

all: $(if $(BUILD_LOCAL_LUAJIT),build-local-luajit) liblua_filter.so

liblua_filter.so: lua_runtime.o
	$(CC) $(LDFLAGS) -o $@ $< $(LUAJIT_LDFLAGS)

lua_runtime.o: lua_runtime.c lua_runtime.h
	$(CC) $(CFLAGS) $(LUAJIT_CFLAGS) -c -o $@ $<

build-local-luajit:
	@echo "Building LuaJIT from source..."
	@if [ ! -d "$(LUAJIT_LOCAL_DIR)" ]; then \
		git clone https://github.com/LuaJIT/LuaJIT.git $(LUAJIT_LOCAL_DIR) && \
		cd $(LUAJIT_LOCAL_DIR) && \
		git checkout v2.1 && \
		make; \
	fi

install-deps:
	@echo "Attempting to install LuaJIT dependencies..."
	@if command -v apt-get >/dev/null; then \
		sudo apt-get update && sudo apt-get install -y luajit libluajit-5.1-dev pkg-config || \
		echo "Could not install with apt-get, will attempt to build from source"; \
	elif command -v yum >/dev/null; then \
		sudo yum install -y luajit luajit-devel pkg-config || \
		echo "Could not install with yum, will attempt to build from source"; \
	else \
		echo "No supported package manager found, will attempt to build from source"; \
	fi
	@if [ "$$?" -ne 0 ]; then \
		$(MAKE) build-local-luajit; \
	fi

test: all
	@echo "Testing LuaJIT URL filter..."
	@LUA_MODULE_PATH="$(PWD)/url_filter.lua" ./test_filter.sh

clean:
	rm -f *.o *.so
	if [ -d "$(LUAJIT_LOCAL_DIR)" ]; then \
		cd $(LUAJIT_LOCAL_DIR) && make clean; \
	fi

.PHONY: all clean install-deps test build-local-luajit 