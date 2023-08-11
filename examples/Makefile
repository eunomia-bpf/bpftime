.PHONY: example
example:
	rm -f example
	g++ -Wl,--verbose -o example example.c -I../runtime/include -I../third_party \
	-lm \
	../build/runtime/libruntime.a \
	../build/libbpftime.a \
	../build/runtime/libbpf/libbpf.a \
	 -lelf -lz  \
	  ../build/FridaGum-prefix/src/FridaGum/libfrida-gum.a -lpthread -ldl -lm 
