.PHONY: install coverage test docs help build clean build-arm run-arm run-arm64 build-arm64 build-arm32
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

JOBS := 1

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

test: build-arm32
	cd build/ && ctest -VV

test-vm: ## run tests quickly with pytest on x86, you need to build first
	pytest -v -s test/test_framework

coverage: ## check code coverage quickly GCC
	rm -rf build/
	cmake -Bbuild -DCMAKE_INSTALL_PREFIX=$(INSTALL_LOCATION) -DBPFTIME_ENABLE_CODE_COVERAGE=1
	cmake --build build --config Release -j$(JOBS)
	cd build/ && ctest -C Release -VV
	cd .. && (bash -c "find . -type f -name '*.gcno' -exec gcov -pb {} +" || true)

build-all: build-all build-arm32 build-arm64

build: ## build the package
	rm -rf build/
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1
	cmake --build build --config Debug  -j$(JOBS)

build-ubpf: ## build the package with ubpf backend
	rm -rf build/
	cmake -Bbuild  -DBPFTIME_BUILD_STANDALONE_VM=YES
	cmake --build build --config Release -j$(JOBS)

build-llvm: ## build with llvm as jit backend
	cmake -Bbuild -DBPFTIME_LLVM_JIT=1 -DBPFTIME_BUILD_STANDALONE_VM=YES
	cmake --build build --config Release -j$(JOBS)

build-arm64: ## build the package on arm64
	rm -rf build/
	cmake -Bbuild -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake -DARCH=aarch64 -DBPFTIME_ENABLE_UNIT_TESTING=1
	cmake --build build --config Debug -j$(JOBS)

run-arm64: build-arm64 ## run the binary on arm32 qemu
	qemu-aarch64 -L /usr/aarch64-linux-gnu/ build/bin/Debug/libebpf

install: ## install the package to the `INSTALL_LOCATION`
	rm -rf build/
	cmake -Bbuild -DCMAKE_INSTALL_PREFIX=$(INSTALL_LOCATION)
	cmake --build build --config Release
	cmake --build build --target install --config Release -j$(JOBS)

format: ## format the project sources
	rm -rf build/
	cmake -Bbuild -DCMAKE_INSTALL_PREFIX=$(INSTALL_LOCATION)
	cmake --build build --target clang-format  -j$(JOBS)

clean: ## clean the project
	rm -rf build/
