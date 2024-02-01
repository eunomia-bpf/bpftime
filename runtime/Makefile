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

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

test: build ## test the package
	# make -C test/example
	cd build/ && ctest -VV

build: ## build the package
	rm -rf build/
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1
	cmake --build build --config Debug

clean: ## clean the project
	rm -rf build/
