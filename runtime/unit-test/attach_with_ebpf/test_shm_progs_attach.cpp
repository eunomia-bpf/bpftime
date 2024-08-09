/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#include "bpftime_ufunc.hpp"
#include "bpftime_shm.hpp"
#include "frida_attach_private_data.hpp"
#include "frida_attach_utils.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "handler/handler_manager.hpp"
#include "handler/link_handler.hpp"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include "bpftime_object.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <bpftime_prog.hpp>
#include <catch2/catch_test_macros.hpp>
#include <unit-test/common_def.hpp>
using namespace boost::interprocess;
using namespace bpftime;
using namespace bpftime::attach;
static const char *HANDLER_NAME = "my_handler";
static const char *SHM_NAME = "my_shm_attach_test";

// This is the original function to hook.
// optnone attribute may be needed to prevent LTO from optimizing the function
extern "C" __attribute__((__noinline__, noinline)) int
_bpftime_test_shm_progs_attach_my_function(int parm1, const char *str, char c)
{
	asm("");
	SPDLOG_INFO("origin func: Args: {}, {}, {}", parm1, str, c);
	volatile int ret = 35;
	return ret;
}

extern "C" __attribute__((__noinline__, noinline)) int
_bpftime_test_shm_progs_attach_my_uprobe_function(int parm1, const char *str,
						  char c)
{
	asm("");
	SPDLOG_INFO("my_uprobe_function: Args: {}, {}, {}", parm1,
		    (const char *)str, c);
	volatile int ret = 35;
	return ret;
}

extern "C" int add_func(int a, int b)
{
	int res = a + b;
	printf("helper-add %d = %d + %d\n", res, a, b);
	return res;
}

extern "C" uint64_t print_func(char *str)
{
	SPDLOG_INFO("Print func: {:x}", (uintptr_t)str);
	printf("helper-print: %s\n", str);
	return 0;
}

static const char *obj_path = TOSTRING(EBPF_PROGRAM_PATH_REPLACE);

static void register_ufunc_for_print_and_add(bpf_attach_ctx *probe_ctx)
{
	ebpf_ufunc_func_info func2 = { "add_func",
				       UFUNC_FN(add_func),
				       UFUNC_TYPE_INT32,
				       { UFUNC_TYPE_INT32, UFUNC_TYPE_INT32 },
				       2,
				       0,
				       false };
	bpftime_ufunc_resolve_from_info(func2,
					&attach::find_function_addr_by_name);

	ebpf_ufunc_func_info func1 = { "print_func",
				       UFUNC_FN(print_func),
				       UFUNC_TYPE_INT64,
				       { UFUNC_TYPE_POINTER },
				       1,
				       0,
				       false };
	bpftime_ufunc_resolve_from_info(func1,
					&attach::find_function_addr_by_name);
}

static int attach_uprobe(bpftime::handler_manager &manager_ref,
			 managed_shared_memory &segment, bpftime_prog *prog,
			 bpf_attach_ctx &ctx)
{
	std::uint64_t offset = 0;
	void *module_base_self = attach::get_module_base_addr("");
	void *my_uprobe_function_addr = attach::find_function_addr_by_name(
		"_bpftime_test_shm_progs_attach_my_uprobe_function");
	offset = (uintptr_t)my_uprobe_function_addr -
		 (uintptr_t)module_base_self;
	SPDLOG_INFO("my_uprobe_function_addr: {:x}, offset: {}",
		    (uintptr_t)my_uprobe_function_addr, offset);
	// attach uprobe
	bpf_perf_event_handler hd(false, offset, -1, "", 0, segment);
	hd.enable();
	manager_ref.set_handler(4, hd, segment);
	auto &prog_handler = std::get<bpf_prog_handler>(manager_ref[0]);
	// the attach fd is 3
	return manager_ref.set_handler_at_empty_slot(bpf_link_handler(0, 4),
						     segment);
}

static int attach_replace(bpftime::handler_manager &manager_ref,
			  managed_shared_memory &segment, bpftime_prog *prog,
			  bpf_attach_ctx &ctx)
{
	void *module_base_self = attach::get_module_base_addr("");
	void *my_function_addr =
		(void *)_bpftime_test_shm_progs_attach_my_function;
	std::uint64_t offset =
		(uintptr_t)my_function_addr - (uintptr_t)module_base_self;
	manager_ref.set_handler(
		0,
		bpf_prog_handler(
			segment, prog->get_insns().data(),
			prog->get_insns().size(), prog->prog_name(),
			(int)bpftime::bpf_prog_type::BPF_PROG_TYPE_UNSPEC),
		segment);
	auto &prog_handler = std::get<bpf_prog_handler>(manager_ref[0]);

	// attach replace
	manager_ref.set_handler(
		3,
		bpf_perf_event_handler(bpf_event_type::BPF_TYPE_UREPLACE,
				       offset, -1, "", segment, true),
		segment);

	// the attach fd is 3
	return manager_ref.set_handler_at_empty_slot(bpf_link_handler(0, 3),
						     segment);
}

static void handle_sub_process()
{
	int res = 0;
	SPDLOG_INFO("Subprocess started, pid = {}", getpid());
	// test for no attach
	res = _bpftime_test_shm_progs_attach_my_function(1, "hello aaa", 'c');
	SPDLOG_INFO("origin func return: {}", res);
	// REQUIRE(res == 35);
	assert(res == 35);
	managed_shared_memory segment(open_only, SHM_NAME);
	auto manager = segment.find<handler_manager>(HANDLER_NAME).first;

	// init the attach ctx
	bpf_attach_ctx ctx;
	ctx.register_attach_impl(
		{ ATTACH_UPROBE, ATTACH_URETPROBE, ATTACH_UPROBE_OVERRIDE,
		  ATTACH_UREPLACE },
		std::make_unique<attach::frida_attach_impl>(),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach_private_data> priv_data =
				std::make_unique<frida_attach_private_data>();
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	register_ufunc_for_print_and_add(&ctx);
	agent_config config;
	config.enable_ufunc_helper_group = true;
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config) == 0);

	// test for attach
	res = _bpftime_test_shm_progs_attach_my_function(1, "hello replace",
							 'c');
	SPDLOG_INFO("Hooked func return: {}", res);

	assert(res == 100);
	_bpftime_test_shm_progs_attach_my_uprobe_function(2, "hello uprobe",
							  'd');

	manager->clear_all(segment);
	segment.destroy_ptr(manager);
	SPDLOG_INFO("Subprocess exited");
	_exit(0);
}

__attribute__((optnone)) TEST_CASE("Test shm progs attach")
{
	spdlog::set_level(spdlog::level::debug);
	SPDLOG_INFO("parent process start");
	bpftime::shm_remove remover(SHM_NAME);

	// The side that creates the mapping
	managed_shared_memory segment(create_only, SHM_NAME, 1 << 20);
	auto manager =
		segment.construct<handler_manager>(HANDLER_NAME)(segment);
	auto &manager_ref = *manager;

	// open the object file
	bpftime_object *obj = bpftime_object_open(obj_path);
	REQUIRE(obj != nullptr);
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);

	// init the attach ctx
	bpf_attach_ctx ctx;
	ctx.register_attach_impl(
		{ ATTACH_UPROBE, ATTACH_URETPROBE, ATTACH_UPROBE_OVERRIDE,
		  ATTACH_UREPLACE },
		std::make_unique<attach::frida_attach_impl>(),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach_private_data> priv_data =
				std::make_unique<frida_attach_private_data>();
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	register_ufunc_for_print_and_add(&ctx);
	int replace_link_id = attach_replace(manager_ref, segment, prog, ctx);
	int uprobe_link_id = attach_uprobe(manager_ref, segment, prog, ctx);

	agent_config config;
	config.enable_ufunc_helper_group = true;
	ctx.init_attach_ctx_from_handlers(manager, config);

	// test for attach replace
	int res = 0;
	res = _bpftime_test_shm_progs_attach_my_function(1, "hello replace",
							 'c');
	// We have to use volatile. If not, LTO will optimize the process to
	// pass arguments..
	volatile int arg1 = 2;
	const char *arg2 = "hello uprobe";
	volatile char arg3 = 'd';
	res = _bpftime_test_shm_progs_attach_my_uprobe_function(arg1, arg2,
								arg3);
	SPDLOG_INFO("Hooked func return: {}", res);
	REQUIRE(ctx.destroy_instantiated_attach_link(replace_link_id) == 0);
	REQUIRE(ctx.destroy_instantiated_attach_link(uprobe_link_id) == 0);

	int pid = fork();
	if (pid == 0) {
		handle_sub_process();
	} else {
		int status;
		int ret = waitpid(pid, &status, 0);
		REQUIRE(ret != -1);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
		SPDLOG_INFO("Subprocess exited, from server");
		REQUIRE(segment.find<handler_manager>(HANDLER_NAME).first ==
			nullptr);
		SPDLOG_INFO("Server exiting..");
		bpftime_object_close(obj);
	}
}
