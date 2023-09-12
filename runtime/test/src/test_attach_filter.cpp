#include <cstdint>
#include <cstring>
#include <assert.h>
#include <inttypes.h>
#include "bpftime.h"
#include "bpftime_object.h"
#include "bpftime_shm.hpp"
#include "test_defs.h"

using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_NO_CREATE;

static void register_ffi_for_print_and_add(bpf_attach_ctx *probe_ctx)
{
	ebpf_ffi_func_info func2 = { "add_func",
				     FFI_FN(add_func),
				     FFI_TYPE_INT32,
				     { FFI_TYPE_INT32, FFI_TYPE_INT32 },
				     2,
				     0,
				     false };
	bpftime_ffi_resolve_from_info(probe_ctx, func2);

	ebpf_ffi_func_info func1 = { "print_func",
				     FFI_FN(print_func),
				     FFI_TYPE_INT64,
				     { FFI_TYPE_POINTER },
				     1,
				     0,
				     false };
	bpftime_ffi_resolve_from_info(probe_ctx, func1);
}

// This is the original function to hook.
int my_function(const char *str, char c, long long parm1)
{
	// buggy code: not check str is NULL
	int i = str[0];
	printf("origin func: Args: %s, %c, %d\n", str, c, i);
	return (int)parm1;
}

const char *obj_path = "./filter.bpf.o";

uint64_t test_pass_param(uint64_t arg1, uint64_t arg2, uint64_t arg3)
{
	char *str = (char *)arg1;
	char c = (char)arg2;
	long long param1 = (long long)arg3;
	// "hello aaa", 'c', 182
	assert(strcmp(str, "hello aaa") == 0);
	assert(c == 'c');
	assert(param1 == 182);
	return 0;
}

int main()
{
	int res = 1;

	// test for no attach
	res = my_function("hello aaa", 'c', 182);
	printf("origin func return: %d\n", res);
	assert(res == 182);

	bpf_attach_ctx probe_ctx;
	bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);
	// get the first program
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	assert(prog);
	// add ffi support
	bpftime_helper_group::get_ffi_helper_group().add_helper_group_to_prog(
		prog);
	bpftime_helper_group::get_kernel_utils_helper_group()
		.add_helper_group_to_prog(prog);
	struct bpftime_helper_info info = {
		.index = 4097,
		.name = "test_pass_param",
		.fn = (void *)test_pass_param,
	};
	res = prog->bpftime_prog_register_raw_helper(info);
	assert(res == 0);

	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	// attach
	int fd = probe_ctx.create_filter((void *)my_function);
	assert(fd >= 0);
	res = probe_ctx.attach_prog(prog, fd);
	assert(res == 0);

	// test for pass filter
	res = my_function("hello aaa", 'c', 182);
	printf("hooked func return: %d\n", res);
	assert(res == 182);

	// test for drop filter
	res = my_function(NULL, 'c', 1);
	printf("hooked func return: %d\n", res);
	assert(res == -22);

	// detach
	res = probe_ctx.detach(prog);
	assert(res == 0);
	res = probe_ctx.destory_attach(fd);
	assert(res == 0);

	// test for no attach
	res = my_function("hello aaa", 'c', 182);
	printf("hooked func return: %d\n", res);
	assert(res == 182);

	bpftime_object_close(obj);

	return 0;
}
