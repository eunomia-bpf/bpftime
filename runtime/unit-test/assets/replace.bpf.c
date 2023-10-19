#include "vmlinux.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_helpers.h"
#include "ffi.bpf.h"

static inline int add_func(int a, int b)
{
	return FFI_CALL_NAME_2("add_func", a, b);
}

static inline uint64_t print_func(char *str)
{
	return FFI_CALL_NAME_1("print_func", str);
}

int BPF_UPROBE(my_function, int parm1, char* str, char c) {
	uint64_t n = print_func(str);
    return n + add_func(parm1, c);
}
