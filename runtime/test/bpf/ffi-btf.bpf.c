/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "ufunc.bpf.h"

struct data {
	int a;
	int b;
};

#define FUNC_ADD 1
#define FUNC_PRINT 2

static inline int add_func(int a, int b)
{
	return UFUNC_CALL_NAME_2("add_func", a, b);
}

static inline uint64_t print_func(char *str)
{
	return UFUNC_CALL_NAME_1("print_func", str);
}

int bpf_main(struct data *d)
{
	// not support global value
	char str[] = "hello";
	// print_func("hello") not support
	uint64_t n = print_func(str);
	int x = (int)n + d->a;
	return add_func(x, 12);
}
