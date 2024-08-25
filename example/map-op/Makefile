CXX = clang

CXXFLAGS = -I../../runtime/include \
           -I../../vm/compat/include \
		   -I../../third_party/spdlog/include \
		   -I../../vm/vm-core/include

SRC = mapop.bpf.c
TGT = main

main: $(SRC)
	$(CXX) $(SRC) -g --target=bpf -o $(TGT) $(CXXFLAGS) 

.PHONY: clean
clean:
	rm -rf $(TGT)