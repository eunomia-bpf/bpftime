// test the ufunc function call with core ebpf vm
#include <string.h>
#include "test_bpf_progs.h"
#include "test_minimal_bpf_host_ufunc.h"
#include <stdlib.h>
#include <inttypes.h>
#include "test_defs.h"

struct ebpf_inst;

#define TEST_BPF_CODE bpf_ufunc_code
#define TEST_BPF_SIZE sizeof(bpf_ufunc_code) - 1

typedef unsigned int (*kernel_fn)(const void *ctx,
				  const struct ebpf_inst *insn);

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

	register_ufunc_handler(vm);

	ebpf_toggle_bounds_check(vm, false);

	// remove 0, in the end
	CHECK_EXIT(ebpf_load(vm, TEST_BPF_CODE, TEST_BPF_SIZE, &errmsg));

	// EBPF_OP_CALL
	printf("code len: %zd\n", TEST_BPF_SIZE);

	// int mem_len = 1024 * 1024;
	// char *mem = (char *) malloc(mem_len);
	CHECK_EXIT(ebpf_exec(vm, &m, sizeof(m), &res));
	printf("res = %" PRIu64 "\n", res);
	return 0;
}