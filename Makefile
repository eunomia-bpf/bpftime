.PHONY: install coverage test docs help build clean unit-test-daemon unit-test unit-test-runtime
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z\d_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"
INSTALL_LOCATION := ~/.local

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build-unit-test:
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1 -DCMAKE_BUILD_TYPE:STRING=Debug
	cmake --build build --config Debug --target bpftime_runtime_tests

unit-test-daemon: 
	build/daemon/test/bpftime_daemon_tests

unit-test-runtime:  ## run catch2 unit tests
	make -C runtime/test/bpf && cp runtime/test/bpf/*.bpf.o build/runtime/test/
	./build/runtime/unit-test/bpftime_runtime_tests
	cd build/runtime/test && ctest -VV

unit-test: unit-test-daemon unit-test-runtime ## run catch2 unit tests

build: ## build the package
	cmake -Bbuild -DBPFTIME_ENABLE_UNIT_TESTING=1
	cmake --build build --config Debug
	cd tools/cli-rs && cargo build

release: ## build the package
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=0 \
				   -DCMAKE_BUILD_TYPE:STRING=Release \
				   -DBPFTIME_ENABLE_LTO=1
	cmake --build build --config Release --target install
	cd tools/cli-rs && cargo build --release

build-vm: ## build only the core library
	make -C vm build

build-llvm: ## build with llvm as jit backend
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1 -DBPFTIME_LLVM_JIT=1
	cmake --build build --config Debug

clean: ## clean the project
	rm -rf build
	make -C runtime clean
	make -C vm clean

install: release ## Invoke cmake to install..
	cd tools/cli-rs && mkdir -p ~/.bpftime && cp ./target/release/bpftime ~/.bpftime
