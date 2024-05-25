CXX = g++

CXXFLAGS = -I../../runtime/include \
           -I../../vm/compat/include \
		   -I../../third_party/spdlog/include \
		   -I../../vm/vm-core/include

.DEFAULT_GOAL := help

LDFLAGS = -L$(HOME)/.bpftime
LDLIBS = -lbpftime -lboost_system -lrt -lbpf

SHM_SRC = main.cpp

SHM_TARGET = shm_example

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf " %-20s %s\n", $$1, $$2}'


shm_example: $(SHM_SRC) ## build shared memory example
	$(CXX) -o $(SHM_TARGET) $(SHM_SRC) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean: ## clean the build
	rm -f $(SHM_TARGET) example_shm.json
