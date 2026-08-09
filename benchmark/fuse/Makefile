.PHONY: build
build: filebench passthrough
	make -C bpf

filebench: bench.cpp
	g++ -o filebench bench.cpp -lpthread

passthrough: fs/passthrough.c
	gcc -o fs/passthrough fs/passthrough.c -Wall -D_FILE_OFFSET_BITS=64 -lfuse -lpthread -I/usr/include/fuse 

clean:
	make -C bpf clean
	rm -f filebench fs/passthrough

