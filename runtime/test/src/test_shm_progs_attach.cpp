/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#include "bpftime_ffi.hpp"
#include "bpftime_shm.hpp"
#include "handler/handler_manager.hpp"
#include <cstdlib>
#include "bpftime_object.hpp"
#include "test_defs.h"
#include <boost/interprocess/creation_tags.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <unistd.h>
#include <bpftime_prog.hpp>
#include "attach/attach_manager/base_attach_manager.hpp"

using namespace boost::interprocess;
using namespace bpftime;

const char *HANDLER_NAME = "my_handler";
const char *SHM_NAME = "my_shm_attach_test";

// This is the original function to hook.
extern "C" int my_function(int parm1, const char *str, char c)
{
	printf("origin func: Args: %d, %s, %c\n", parm1, str, c);
	return 35;
}

extern "C" int my_uprobe_function(int parm1, const char *str, char c)
{
	printf("my_uprobe_function: Args: %d, %s, %c\n", parm1, str, c);
	return 35;
}

const char *obj_path = "./replace.bpf.o";

static void register_ffi_for_print_and_add(bpf_attach_ctx *probe_ctx)
{
	ebpf_ffi_func_info func2 = { "add_func",
				     FFI_FN(add_func),
				     FFI_TYPE_INT32,
				     { FFI_TYPE_INT32, FFI_TYPE_INT32 },
				     2,
				     0,
				     false };
	bpftime_ffi_resolve_from_info(&probe_ctx->get_attach_manager(), func2);

	ebpf_ffi_func_info func1 = { "print_func",
				     FFI_FN(print_func),
				     FFI_TYPE_INT64,
				     { FFI_TYPE_POINTER },
				     1,
				     0,
				     false };
	bpftime_ffi_resolve_from_info(&probe_ctx->get_attach_manager(), func1);
}

void attach_uprobe(bpftime::handler_manager &manager_ref,
		   managed_shared_memory &segment, bpftime_prog *prog,
		   bpf_attach_ctx &ctx)
{
	std::uint64_t offset = 0;
	void *module_base_self =
		ctx.get_attach_manager().get_module_base_addr("");
	void *my_uprobe_function_addr =
		ctx.get_attach_manager().find_function_addr_by_name(
			"my_uprobe_function");
	offset = (uintptr_t)my_uprobe_function_addr -
		 (uintptr_t)module_base_self;
	printf("my_uprobe_function_addr: %p, offset: %lu\n",
	       my_uprobe_function_addr, offset);
	// attach uprobe
	manager_ref.set_handler(
		4, bpf_perf_event_handler(false, offset, -1, "", 0, segment),
		segment);
	auto &prog_handler = std::get<bpf_prog_handler>(manager_ref[0]);
	// the attach fd is 3
	prog_handler.add_attach_fd(4);
}

void attach_replace(bpftime::handler_manager &manager_ref,
		    managed_shared_memory &segment, bpftime_prog *prog,
		    bpf_attach_ctx &ctx)
{
	void *module_base_self =
		ctx.get_attach_manager().get_module_base_addr("");
	void *my_function_addr = (void *)my_function;
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
	// the attach fd is 3
	prog_handler.add_attach_fd(3);
	// attach replace
	manager_ref.set_handler(
		3,
		bpf_perf_event_handler(bpf_event_type::BPF_TYPE_UREPLACE, offset,
				       -1, "", segment, true),
		segment);
}

int main(int argc, const char **argv)
{
	spdlog::set_level(spdlog::level::debug);
	if (argc == 1) {
		std::cout << "parent process start" << std::endl;
		shm_remove remover(SHM_NAME);

		// The side that creates the mapping
		managed_shared_memory segment(create_only, SHM_NAME, 1 << 20);
		auto manager = segment.construct<handler_manager>(HANDLER_NAME)(
			segment);
		auto &manager_ref = *manager;

		// open the object file
		bpftime_object *obj = bpftime_object_open(obj_path);
		assert(obj);
		bpftime_prog *prog = bpftime_object__next_program(obj, NULL);

		// init the attach ctx
		bpf_attach_ctx ctx;
		attach_replace(manager_ref, segment, prog, ctx);
		attach_uprobe(manager_ref, segment, prog, ctx);

		register_ffi_for_print_and_add(&ctx);
		agent_config config;
		config.enable_ffi_helper_group = true;
		ctx.init_attach_ctx_from_handlers(manager, config);

		// test for attach replace
		int res = my_function(1, "hello replace", 'c');
		printf("hooked func return: %d\n", res);
		assert(res == 100);
		my_uprobe_function(2, "hello uprobe", 'd');

		std::cout << "Starting sub process" << std::endl;
		system((std::string(argv[0]) + " sub").c_str());
		std::cout << "Subprocess exited, from server" << std::endl;
		assert(segment.find<handler_manager>(HANDLER_NAME).first ==
		       nullptr);
		std::cout << "Server exiting.." << std::endl;
		bpftime_object_close(obj);
	} else {
		int res = 0;
		std::cout << "Subprocess started, pid=" << getpid()
			  << std::endl;

		// test for no attach
		res = my_function(1, "hello aaa", 'c');
		printf("origin func return: %d\n", res);
		assert(res == 35);

		managed_shared_memory segment(open_only, SHM_NAME);
		auto manager =
			segment.find<handler_manager>(HANDLER_NAME).first;

		// init the attach ctx
		bpf_attach_ctx ctx;
		register_ffi_for_print_and_add(&ctx);
		agent_config config;
		config.enable_ffi_helper_group = true;
		ctx.init_attach_ctx_from_handlers(manager, config);

		// test for attach
		res = my_function(1, "hello replace", 'c');
		printf("hooked func return: %d\n", res);
		assert(res == 100);
		my_uprobe_function(2, "hello uprobe", 'd');

		manager->clear_all(segment);
		std::cout << "Print maps value finished" << std::endl;
		segment.destroy_ptr(manager);
		std::cout << "Subprocess exited" << std::endl;
	}
	return 0;
}
