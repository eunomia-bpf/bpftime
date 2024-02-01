// test jit for vm
#include <string.h>
#include "test_bpf_progs.h"
#include "test_defs.h"
#include "ebpf-vm.h"
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h> 

struct ebpf_inst;

#define TEST_BPF_CODE bpf_add_mem_64_bit
#define TEST_BPF_SIZE sizeof(bpf_add_mem_64_bit)

typedef unsigned int (*kernel_fn)(const void *ctx, const struct ebpf_inst *insn);

char *errmsg;
struct mem {
	uint64_t val;
};

int main()
{
	struct mem m = { __LINE__ };
	uint64_t res = 0;
	// using ubpf jit for x86_64 and arm64
	struct ebpf_vm *vm = ebpf_create();

	ebpf_toggle_bounds_check(vm, false);

	// remove 0, in the end
	CHECK_EXIT(ebpf_load(vm, TEST_BPF_CODE, TEST_BPF_SIZE, &errmsg));

	// EBPF_OP_CALL
	printf("code len: %zd\n", TEST_BPF_SIZE);

	int mem_len = 1024 * 1024;
	char *mem = malloc(mem_len);
	printf("Use JIT Mode\n");

	ebpf_jit_fn fn = ebpf_compile(vm, &errmsg);
	if (fn == NULL) {
		fprintf(stderr, "Failed to compile: %s\n", errmsg);
		free(mem);
		return 1;
	}

	// res = ((kernel_fn)(fn))(NULL, context->vm->insnsi);
	res = fn(&m, sizeof(m));
	printf("res = %" PRIu64 "\n", res);
	return 0;
}