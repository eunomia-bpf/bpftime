.PHONY: build example
build: ubpf/build/lib/libubpf.a
	mkdir -p build/include
	cp -r ubpf/build/* build/
	cp -r ubpf/vm/inc/ubpf.h build/include/core-bpf.h
	cp build/vm/ubpf_config.h build/include/
	mv build/lib/libubpf.a build/lib/libcorebpf.a

ubpf/build/lib/libubpf.a:
	git clone https://github.com/iovisor/ubpf
	cd ubpf && cmake -S . -B build -DUBPF_ENABLE_TESTS=true
	cd ubpf && cmake --build build --config Debug

example: build
	gcc -Ibuild/include -Lbuild/lib -lcorebpf example/main.c -o ubpf-example