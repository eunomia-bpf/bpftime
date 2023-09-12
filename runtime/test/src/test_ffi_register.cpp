// test register raw function and call it from ffi
#include "test_defs.h"
#include "bpftime.h"
#include "bpftime_object.h"
#include "bpftime_shm.hpp"
#include <stdio.h>
#include <stdarg.h>

using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_NO_CREATE;

int test_register_ffi()
{
	struct ebpf_ffi_func_info func2 = { "add_func",
					    FFI_FN(add_func),
					    FFI_TYPE_INT32,
					    { FFI_TYPE_INT32, FFI_TYPE_INT32 },
					    2,
					    2,
					    false };
	bpftime_ffi_register_ffi(1, func2);

	struct ebpf_ffi_func_info func1 = { "print_func",
					    FFI_FN(print_func),
					    FFI_TYPE_INT64,
					    { FFI_TYPE_POINTER },
					    1,
					    2,
					    false };
	bpftime_ffi_register_ffi(2, func1);
	return 0;
}

int main(int argc, char **argv)
{
	const char *obj_path = NULL;
	uint64_t return_val;

	// use a struct as memory
	struct data memory = { 5, 2 };

	obj_path = "./ffi.bpf.o";

	test_register_ffi();

	// open the object file
	bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	assert(prog);
	// use the first program and relocate based on btf if btf has been
	// loaded
	int res = bpftime_helper_group::get_ffi_helper_group()
			  .add_helper_group_to_prog(prog);

	assert(res == 0);

	// test vm
	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	return_val = 0;
	res = prog->bpftime_prog_exec(&memory, sizeof(memory), &return_val);
	assert(return_val == 17);
	prog->bpftime_prog_unload();

	// test jit
	res = prog->bpftime_prog_load(true);
	assert(res == 0);
	return_val = 0;
	res = prog->bpftime_prog_exec(&memory, sizeof(memory), &return_val);
	assert(return_val == 17);

	bpftime_object_close(obj);
	return res;
}
