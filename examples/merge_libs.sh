
lib=libpatch.a

ranlib ../build/runtime/libruntime.a 
ranlib ../build/runtime/libruntime.a 
ranlib ../build/core/simple-jit/libcore-bpf.a 
ranlib ../build/runtime/libbpf/libbpf.a 
ranlib ../build/FridaGum-prefix/src/FridaGum/libfrida-gum.a

ar rcs ../build/${lib} ../build/runtime/libruntime.a \
	../build/runtime/libruntime.a \
	../build/core/simple-jit/libcore-bpf.a \
	../build/runtime/libbpf/libbpf.a \
	../build/FridaGum-prefix/src/FridaGum/libfrida-gum.a

ranlib ../build/${lib}
