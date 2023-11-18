/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef TEST_DEF_BPFTIME_H
#define TEST_DEF_BPFTIME_H

#include <stdio.h>
#include <assert.h>
#include <cstdint>

#define CHECK_EXIT(ret)                                                        \
	if (ret != 0) {                                                        \
		fprintf(stderr, "Failed to load code: %s\n", errmsg);          \
		return -1;                                                     \
	}

#define generate_nm_offset_command(name) "nm ./"  name "_Tests | grep ' T ' > " name ".off.txt"
#define generate_btf_command(name) "pahole --btf_encode_force --btf_encode_detached " name ".btf ./" name "_Tests"

#define generate_tests_btf_offset(name) \
{\
	int res = 1; \
	res = system(nm_offset_command(name)); \
	assert(res == 0); \
	system(generate_btf_command(name)); \
	assert(res == 0); \
}\


struct data {
	int a;
	int b;
} __attribute__((__packed__));

#define DISABLE_RELO 1

struct data_relo {
#if DISABLE_RELO
	int a;
#endif
	int b;
#if DISABLE_RELO
	int c;
#endif
	int d;
};

// avoid const emit
extern "C" int add_func(int a, int b)
{
	int res = a + b;
	printf("helper-add %d = %d + %d\n", res, a, b);
	return res;
}

extern "C" uint64_t print_func(char *str)
{
	printf("helper-print: %s\n", str);
	return 0;
}

#endif
