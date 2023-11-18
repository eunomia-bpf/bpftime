/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef FFI_HELPER_H
#define FFI_HELPER_H

typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef int int32_t;
// global context not support
// uint64_t context;

union arg_val {
	uint64_t uint64;
	int64_t int64;
	int32_t int32;
	double double_val;
	void *ptr;
};

struct arg_list {
	uint64_t args[6];
};

#define FFI_HELPER_ID_DISPATCHER 1000
#define FFI_HELPER_ID_FIND_ID 1001

static const uint64_t (*ffi_call)(uint64_t id, uint64_t arg_list) = (void *)FFI_HELPER_ID_DISPATCHER;
static const uint64_t (*ffi_find_func_id)(const char* func_name) = (void *)FFI_HELPER_ID_FIND_ID;

// func: function id
#define FFI_CALL_0(func) \
    ({ \
        struct arg_list argn = {0}; \
        ffi_call(func, (uint64_t)&argn); \
    })

// func: function id
#define FFI_CALL_1(func, arg1) \
    ({ \
        struct arg_list argn = {0}; \
        argn.args[0] = (uint64_t)arg1; \
        ffi_call(func, (uint64_t)&argn); \
    })

// func: function id
#define FFI_CALL_2(func, arg1, arg2) \
    ({ \
	struct arg_list argn = {0}; \
	argn.args[0] = (uint64_t)arg1; \
	argn.args[1] = (uint64_t)arg2; \
	ffi_call(func, (uint64_t)&argn); \
    })

// func: function id
#define FFI_CALL_3(func, arg1, arg2, arg3) \
    ({ \
	struct arg_list argn = {0}; \
	argn.args[0] = (uint64_t)arg1; \
	argn.args[1] = (uint64_t)arg2; \
	argn.args[2] = (uint64_t)arg3; \
	ffi_call(func, (uint64_t)&argn); \
    })

// func: function name
#define FFI_CALL_NAME_0(func_name) \
    ({ \
		char funcname[] = func_name; \
		uint64_t func_id = ffi_find_func_id(funcname); \
		FFI_CALL_0(func_id);\
    })


#define FFI_CALL_NAME_1(func_name, arg1) \
    ({ \
		char funcname[] = func_name; \
		uint64_t func_id = ffi_find_func_id(funcname); \
		FFI_CALL_1(func_id, arg1);\
    })

#define FFI_CALL_NAME_2(func_name, arg1, arg2) \
    ({ \
		char funcname[] = func_name; \
		uint64_t func_id = ffi_find_func_id(funcname); \
		FFI_CALL_2(func_id, arg1, arg2); \
    })

#define FFI_CALL_NAME_3(func_name, arg1, arg2, arg3) \
    ({ \
		char funcname[] = func_name; \
		uint64_t func_id = ffi_find_func_id(funcname); \
		FFI_CALL_3(func_id, arg1, arg2, arg3); \
    })

#endif