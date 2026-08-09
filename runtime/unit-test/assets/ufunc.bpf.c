/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "ufunc.bpf.h"

struct data {
	int a;
	int b;
} __attribute__((__packed__));

#define FUNC_ADD 1
#define FUNC_PRINT 2

static inline int add_func(int a, int b)
{
	return UFUNC_CALL_2(FUNC_ADD, a, b);
}

static inline uint64_t print_func(char *str)
{
	return UFUNC_CALL_1(FUNC_PRINT, str);
}

#define SEC(name) _Pragma("GCC diagnostic push") \
	_Pragma("GCC diagnostic ignored \"-Wignored-attributes\"") \
	__attribute__((section(name), used)) \
	_Pragma("GCC diagnostic pop")

SEC("uprobe")
int bpf_main(struct data *d)
{
	// not support global value
	char str[] = "hello";
	// print_func("hello") not support
	uint64_t n = print_func(str);
	int x = (int)n + d->a;
	return add_func(x, 12);
}
