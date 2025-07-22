CXX = g++

CXXFLAGS = -I../../runtime/include \
           -I../../vm/compat/include \
		   -I../../third_party/spdlog/include \
		   -I../../vm/vm-core/include

SRC = main.cpp
TGT = main

main: $(SRC)
	$(CXX) $(SRC) -o $(TGT) $(CXXFLAGS) 

.PHONY: clean
clean:
	rm -rf $(TGT)