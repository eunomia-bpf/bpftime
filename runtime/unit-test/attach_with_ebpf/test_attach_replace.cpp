/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <stdio.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "bpftime_ufunc.hpp"
#include <unit-test/common_def.hpp>
#include <catch2/catch_test_macros.hpp>
#include <bpftime_object.hpp>
#include <bpftime_helper_group.hpp>
#include <frida_uprobe.hpp>
extern "C" uint64_t bpftime_set_retval(uint64_t value);

using namespace bpftime;

extern "C" int _bpftime_test_attach_replace__add_func(int a, int b)
{
	int res = a + b;
	printf("helper-add %d = %d + %d\n", res, a, b);
	return res;
}

extern "C" uint64_t _bpftime_test_attach_replace__print_func(char *str)
{
	printf("helper-print: %s\n", str);
	return 0;
}

// This is the original function to hook.
extern "C" __attribute__((optnone)) int
_bpftime_test_attach_replace__my_function(int parm1, const char *str, char c)
{
	spdlog::info("origin func: Args: {}, {}, {}", parm1, str, c);
	volatile int ret = 35;
	return ret;
}

static const char *obj_path = TOSTRING(EBPF_PROGRAM_PATH_REPLACE);

static void register_ufunc_for_print_and_add(bpf_attach_ctx *probe_ctx)
{
	ebpf_ufunc_func_info func2 = {
		"add_func",
		UFUNC_FN(_bpftime_test_attach_replace__add_func),
		UFUNC_TYPE_INT32,
		{ UFUNC_TYPE_INT32, UFUNC_TYPE_INT32 },
		2,
		0,
		false
	};
	bpftime_ufunc_resolve_from_info(func2,
					&attach::find_function_addr_by_name);

	ebpf_ufunc_func_info func1 = {
		"print_func",
		UFUNC_FN(_bpftime_test_attach_replace__print_func),
		UFUNC_TYPE_INT64,
		{ UFUNC_TYPE_POINTER },
		1,
		0,
		false
	};
	bpftime_ufunc_resolve_from_info(func1,
					&attach::find_function_addr_by_name);
}
__attribute__((optnone)) TEST_CASE("Test attach replace with ebpf")
{
	int res = 1;

	// test for no attach
	res = _bpftime_test_attach_replace__my_function(1, "hello aaa", 'c');
	spdlog::info("origin func return: {}", res);
	REQUIRE(res == 35);

	bpf_attach_ctx probe_ctx;
	register_ufunc_for_print_and_add(&probe_ctx);
	bpftime_object *obj = bpftime_object_open(obj_path);
	REQUIRE(obj != nullptr);
	// get the first program
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	REQUIRE(prog != nullptr);
	// add ufunc support
	res = bpftime_helper_group::get_ufunc_helper_group()
		      .add_helper_group_to_prog(prog);
	REQUIRE(res == 0);
	res = prog->bpftime_prog_load(false);
	REQUIRE(res == 0);
	attach::frida_attach_impl man;
	// attach
	int fd = man.create_uprobe_override_at(
		(void *)_bpftime_test_attach_replace__my_function,
		[=](const pt_regs &regs) {
			uint64_t ret;
			prog->bpftime_prog_exec((void *)&regs, sizeof(regs),
						&ret);
			bpftime_set_retval(ret);
		});
	REQUIRE(fd >= 0);

	// test for attach
	res = _bpftime_test_attach_replace__my_function(1, "hello aaa", 'c');
	spdlog::info("hooked func return: {}", res);
	REQUIRE(res == 100);

	// detach
	res = man.detach_by_id(fd);
	REQUIRE(res == 0);

	// test for no attach
	res = _bpftime_test_attach_replace__my_function(1, "hello aaa", 'c');
	spdlog::info("origin func return: {}", res);
	REQUIRE(res == 35);

	bpftime_object_close(obj);
}
