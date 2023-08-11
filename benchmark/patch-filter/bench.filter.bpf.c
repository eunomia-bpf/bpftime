#define BPF_NO_GLOBAL_DATA
#include "vmlinux.h"
#include "bpf_tracing.h"
#include "bpf_helpers.h"
#include "ffi.bpf.h"

enum hotpatch_op {
	OP_SKIP,
	OP_RESUME,
};

SEC("uprobe//home/yunwei/bpftime/build/tools/cli/bpftime-cli:__benchmark_test_function")
int BPF_UPROBE(__benchmark_test_function, const char *a, int b, uint64_t c) {
    // bpf_printk("__benchmark_test_function: %s %d\n", a, b);

    // Disallow sourcing a file in the sandbox, the commands would be
    // executed later, possibly outside of the sandbox.
    if (a == NULL) {
        return OP_SKIP;
    }
    return OP_RESUME;
}