// test register raw function and call it from ffi
#include "test_defs.h"
#include "bpftime.hpp"
#include "bpftime_object.h"
#include <stdio.h>
#include <stdarg.h>
#include "bpftime_shm.hpp"

using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type =
	shm_open_type::SHM_NO_CREATE;

static void dump_type(void *ctx, const char *fmt, va_list args)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	vprintf(fmt, args);
#pragma GCC diagnostic pop
}

int main(int argc, char **argv)
{
	const char *obj_path = NULL;

	// use a struct as memory
	struct data memory = { 5, 2 };

	obj_path = "./helpers.bpf.o";

	// open the object file
	bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);
	bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	assert(prog);
	// use the first program and relocate based on btf if btf has been
	// loaded
	int res = bpftime_helper_group::get_kernel_utils_helper_group()
			  .add_helper_group_to_prog(prog);

	assert(res == 0);
	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	uint64_t return_val = 0;
	res = prog->bpftime_prog_exec(&memory, sizeof(memory), &return_val);
	assert(return_val == 23);
	bpftime_object_close(obj);
	return res;
}
