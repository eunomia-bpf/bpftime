/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
typedef unsigned long long uint64_t;

enum PatchOp {
	OP_SKIP,
	OP_RESUME,
};

struct patch_args {
	void *ctx; // frida-context
	uint64_t arg0;
	uint64_t arg1;
	uint64_t arg2;
	uint64_t arg3;
	uint64_t arg4;
	uint64_t arg5;
	// arg6-argN danymically get from stack
	
	uint64_t return_val;
} __attribute__((packed));

#define PATCH_HELPER_ID_DISPATCHER 2000

static const uint64_t (*bpftime_bpf_get_args)(struct patch_args *args, int n) = (void *)PATCH_HELPER_ID_DISPATCHER;

static inline int get_arg_n(struct patch_args *args, int n) {
	switch (n) {
	case 0:
		return args->arg0;
	case 1:
		return args->arg1;
	case 2:
		return args->arg2;
	case 3:
		return args->arg3;
	case 4:
		return args->arg4;
	case 5:
		return args->arg5;
	default:
		return bpftime_bpf_get_args(args, n);
		// return n + 1;
	}
	return 0;
}

typedef long long int64_t;

// test replace
int64_t my_add(struct patch_args *args) {
	int64_t a1 = get_arg_n(args, 0);
	int64_t a2 = get_arg_n(args, 1);
	int64_t a3 = get_arg_n(args, 2);
	int64_t a4 = get_arg_n(args, 3);
	int64_t a5 = get_arg_n(args, 4);
	int64_t a6 = get_arg_n(args, 5);
	int64_t a7 = get_arg_n(args, 6);
	int64_t a8 = get_arg_n(args, 7);

	args->return_val = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + 1;

	if (args->return_val < 40) {
		return OP_SKIP;
	} else if (args->return_val < 100) {
		return OP_SKIP;
	}

	return OP_RESUME;
}