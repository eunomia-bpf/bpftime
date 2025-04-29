.PHONY: build
build: filebench
	make -C bpf

filebench: bench.cpp
	g++ -o filebench bench.cpp -lpthread

clean:
	make -C bpf clean

