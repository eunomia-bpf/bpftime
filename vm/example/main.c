#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h> 
#include "bpf_progs.h"
#include "ebpf-vm.h"

struct ebpf_inst;

#define JIT_TEST_UBPF 1

#define TEST_BPF_CODE bpf_div64_code
#define TEST_BPF_SIZE sizeof(bpf_div64_code) - 1

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
	res = ebpf_load(vm, TEST_BPF_CODE, TEST_BPF_SIZE, &errmsg);
	if (res != 0) {
		fprintf(stderr, "Failed to load: %s\n", errmsg);
		return 1;
	}

	// EBPF_OP_CALL
	printf("code len: %zd\n", TEST_BPF_SIZE);

	res = ebpf_exec(vm, &m, sizeof(m), &res);
	if (res != 0) {
		fprintf(stderr, "Failed to exec: %s\n", errmsg);
		return 1;
	}
	printf("res = %" PRIu64 "\n", res);
	return 0;
}
