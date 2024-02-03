/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// test register raw function and call it from ufunc
#include "bpftime_helper_group.hpp"
#include "bpftime_object.hpp"
#include "unit-test/common_def.hpp"
#include <stdio.h>
#include <stdarg.h>
#include <catch2/catch_test_macros.hpp>


using namespace bpftime;

struct data {
	int a;
	int b;
} __attribute__((__packed__));

static void dump_type(void *ctx, const char *fmt, va_list args)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	vprintf(fmt, args);
#pragma GCC diagnostic pop
}

TEST_CASE("Test helpers")
{
	const char *obj_path = NULL;

	// use a struct as memory
	struct data memory = { 5, 2 };

	obj_path = TOSTRING(EBPF_PROGRAM_PATH_HELPERS);

	// open the object file
	bpftime_object *obj = bpftime_object_open(obj_path);
	REQUIRE(obj != nullptr);
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	REQUIRE(prog != nullptr);
	// use the first program and relocate based on btf if btf has been
	// loaded
	int res = bpftime_helper_group::get_kernel_utils_helper_group()
			  .add_helper_group_to_prog(prog);

	REQUIRE(res == 0);
	res = prog->bpftime_prog_load(false);
	REQUIRE(res == 0);
	uint64_t return_val = 0;
	res = prog->bpftime_prog_exec(&memory, sizeof(memory), &return_val);
	REQUIRE(return_val == 23);
	bpftime_object_close(obj);
}
