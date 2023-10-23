#include <stdio.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <assert.h>
#include <inttypes.h>
#include "bpftime.hpp"
#include "bpftime_object.hpp"
#include "test_defs.h"
#include "bpftime_shm.hpp"
#include "bpftime_ffi.hpp"

using namespace bpftime;

// This is the original function to hook.
int my_function(int parm1, const char *str, char c)
{
	printf("origin func: Args: %d, %s, %c\n", parm1, str, c);
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

int main()
{
	int res = 1;

	// test for no attach
	res = my_function(1, "hello aaa", 'c');
	printf("origin func return: %d\n", res);
	assert(res == 35);

	bpf_attach_ctx probe_ctx;
	register_ffi_for_print_and_add(&probe_ctx);
	bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);
	// get the first program
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	assert(prog);
	// add ffi support
	res = bpftime_helper_group::get_ffi_helper_group()
		      .add_helper_group_to_prog(prog);
	assert(res == 0);
	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	// attach
	int fd = probe_ctx.get_attach_manager().attach_replace_at(
		(void *)my_function, [=](const pt_regs &regs) -> uint64_t {
			uint64_t ret;
			prog->bpftime_prog_exec((void *)&regs, sizeof(regs),
						&ret);
			return ret;
		});
	assert(fd >= 0);

	// test for attach
	res = my_function(1, "hello aaa", 'c');
	printf("hooked func return: %d\n", res);
	assert(res == 100);

	// detach
	res = probe_ctx.get_attach_manager().destroy_attach(fd);
	assert(res == 0);

	// test for no attach
	res = my_function(1, "hello aaa", 'c');
	printf("origin func return: %d\n", res);
	assert(res == 35);

	bpftime_object_close(obj);

	return 0;
}
