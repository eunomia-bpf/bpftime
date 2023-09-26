#ifndef LLVM_JIT_H
#define LLVM_JIT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "ebpf-vm.h"
#include "ebpf_inst.h"
#include <ebpf-vm.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 0

// Add a log function using DEBUG and printf
#define LOG_DEBUG(...)                                                         \
	do {                                                                   \
		if (DEBUG)                                                     \
			printf(__VA_ARGS__);                                   \
	} while (0)

typedef uint64_t (*ext_func)(uint64_t arg0, uint64_t arg1, uint64_t arg2,
			     uint64_t arg3, uint64_t arg4);

struct bpf_jit_context;

struct ebpf_vm {
	/* ubpf_defs*/
	/* Instructions for interpreter */
	struct ebpf_inst *insnsi = NULL;
	uint16_t num_insts;
	bool bounds_check_enabled;
	ext_func ext_funcs[MAX_EXT_FUNCS];
	const char **ext_func_names = NULL;
	int unwind_stack_extension_index;
	int (*error_printf)(FILE *stream, const char *format, ...) = NULL;
	uint64_t pointer_secret;
	ebpf_jit_fn jitted_function;
	bpf_jit_context *jit_context;
	uint64_t (*map_by_fd)(uint32_t);
	uint64_t (*map_by_idx)(uint32_t);
	uint64_t (*map_val)(uint64_t);
	uint64_t (*var_addr)(uint32_t);
	uint64_t (*code_addr)(uint32_t);
};

#ifdef __cplusplus
}
#endif

#endif // CORE_EBPF_VM_H
