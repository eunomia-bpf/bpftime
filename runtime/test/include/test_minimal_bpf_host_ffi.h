/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPF_MINIMAL_TEST_UFUNC_DEFS_H
#define BPF_MINIMAL_TEST_UFUNC_DEFS_H

#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "ebpf-vm.h"

#define MAX_UFUNC_FUNCS 128
#define MAX_ARGS 5

/* Useful for eliminating compiler warnings.  */
#define UFUNC_FN(f) ((void *)((void (*)(void))f))

enum ffi_types {
	UFUNC_TYPE_VOID,
	UFUNC_TYPE_INT,
	UFUNC_TYPE_UINT,
	UFUNC_TYPE_LONG,
	UFUNC_TYPE_ULONG,
	UFUNC_TYPE_FLOAT,
	UFUNC_TYPE_DOUBLE,
	UFUNC_TYPE_POINTER,
	UFUNC_TYPE_STRUCT,
	UFUNC_TYPE_STRING,
	UFUNC_TYPE_BOOL,
	UFUNC_TYPE_INT8,
	UFUNC_TYPE_UINT8,
	UFUNC_TYPE_INT16,
	UFUNC_TYPE_UINT16,
	UFUNC_TYPE_INT32,
	UFUNC_TYPE_UINT32,
	UFUNC_TYPE_INT64,
	UFUNC_TYPE_UINT64,
	UFUNC_TYPE_INT128,
	UFUNC_TYPE_UINT128,
	UFUNC_TYPE_ENUM,
	UFUNC_TYPE_ARRAY,
	UFUNC_TYPE_UNION,
	UFUNC_TYPE_FUNCTION,
};

static struct ebpf_ffi_func_info *ebpf_resovle_ffi_func(uint64_t func_id);

typedef void *(*ffi_func)(void *r1, void *r2, void *r3, void *r4, void *r5);

struct ebpf_ffi_func_info {
	ffi_func func;
	enum ffi_types ret_type;
	enum ffi_types arg_types[MAX_ARGS];
	int num_args;
};

struct arg_list {
	uint64_t args[6];
};

union arg_val {
	uint64_t uint64;
	int64_t int64;
	double double_val;
	void *ptr;
};

static inline union arg_val to_arg_val(enum ffi_types type, uint64_t val)
{
	union arg_val arg;
	switch (type) {
	case UFUNC_TYPE_INT:
	case UFUNC_TYPE_UINT:
		arg.uint64 = val;
		break;
	case UFUNC_TYPE_DOUBLE:
		arg.double_val = *(double *)&val;
		break;
	case UFUNC_TYPE_POINTER:
		arg.ptr = (void *)(uintptr_t)val;
		break;
	default:
		// Handle other types
		break;
	}
	return arg;
}

static inline uint64_t from_arg_val(enum ffi_types type, union arg_val val)
{
	switch (type) {
	case UFUNC_TYPE_INT:
	case UFUNC_TYPE_UINT:
		return val.uint64;
	case UFUNC_TYPE_DOUBLE:
		return *(uint64_t *)&val.double_val;
	case UFUNC_TYPE_POINTER:
		return (uint64_t)(uintptr_t)val.ptr;
	default:
		// Handle other types
		break;
	}
	return 0;
}

static uint64_t __ebpf_call_ffi_dispatcher(uint64_t id, uint64_t arg_list)
{
	assert(id < MAX_UFUNC_FUNCS);
	struct ebpf_ffi_func_info *func_info = ebpf_resovle_ffi_func(id);
	assert(func_info->func != NULL);

	// Prepare arguments
	struct arg_list *raw_args = (struct arg_list *)(uintptr_t)arg_list;
	union arg_val args[5];
	for (int i = 0; i < func_info->num_args; i++) {
		args[i] =
			to_arg_val(func_info->arg_types[i], raw_args->args[i]);
	}

	// Call the function
	union arg_val ret;
	switch (func_info->num_args) {
	case 0:
		ret.uint64 = func_info->func(0, 0, 0, 0, 0);
		break;
	case 1:
		ret.uint64 = func_info->func(args[0].uint64, 0, 0, 0, 0);
		break;
	case 2:
		ret.uint64 = func_info->func(args[0].uint64, args[1].uint64, 0,
					     0, 0);
		break;
	case 3:
		ret.uint64 = func_info->func(args[0].uint64, args[1].uint64,
					     args[2].uint64, 0, 0);
		break;
	case 4:
		ret.uint64 = func_info->func(args[0].uint64, args[1].uint64,
					     args[2].uint64, args[3].uint64, 0);
		break;
	case 5:
		ret.uint64 = func_info->func(args[0].uint64, args[1].uint64,
					     args[2].uint64, args[3].uint64,
					     args[4].uint64);
		break;
	default:
		// Handle other cases
		break;
	}

	// Convert the return value to the correct type
	return from_arg_val(func_info->ret_type, ret);
}

static uint64_t print_func(char *str)
{
	printf("helper-1: %s\n", str);
	return strlen(str);
}

static int add_func(int a, int b)
{
	return a + b;
}

/* temperially used for test.
Should be implemented via resolvering the function via function name from BTF
symbols
*/
struct ebpf_ffi_func_info func_list[] = {
	{ NULL, UFUNC_TYPE_INT, { UFUNC_TYPE_POINTER }, 1 },
	{ UFUNC_FN(add_func),
	  UFUNC_TYPE_INT,
	  { UFUNC_TYPE_INT, UFUNC_TYPE_INT },
	  2 },
	{ UFUNC_FN(print_func), UFUNC_TYPE_ULONG, { UFUNC_TYPE_POINTER }, 1 },
};

static struct ebpf_ffi_func_info *ebpf_resovle_ffi_func(uint64_t func_id)
{
	const int N_FUNC =
		sizeof(func_list) / sizeof(struct ebpf_ffi_func_info);
	if (func_id < N_FUNC) {
		return &func_list[func_id];
	}
	assert(0);
	return NULL;
}

static void register_ffi_handler(struct ebpf_vm *vm)
{
	ebpf_register(vm, 1, "__ebpf_call_ffi_dispatcher",
		      __ebpf_call_ffi_dispatcher);
}

// static struct bpftime_prog *bpftime_create_context(void)
// {
// 	struct bpftime_prog *prog =
// 		(struct bpftime_prog *)malloc(sizeof(struct bpftime_prog));
// 	struct ebpf_vm *vm = ebpf_create();
// 	context->vm = vm;
// 	ebpf_register(context->vm, 1, "__ebpf_call_ffi_dispatcher",
// 		      __ebpf_call_ffi_dispatcher);
// 	return context;
// }

#endif
