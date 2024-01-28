/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// test find function address, register all functions from btf and call it from
// ebpf
#include "test_defs.h"
#include "bpftime.hpp"
#include "bpftime_object.h"
#include "bpftime_shm.hpp"
#include <cstdint>

const char *offset_record_path = "./test_ufunc.off.txt";
const char *btf_path = "./base.btf";
const char *obj_path = "./ufunc-btf.bpf.o";

int main(int argc, char **argv)
{
	int res = 1;
	uint64_t return_val;

	// use a struct as memory
	struct data memory = { 5, 2 };

	struct bpf_attach_ctx *probe_ctx = bpftime_probe_create_ctx();
	res = bpftime_ufunc_ctx_from_btf(probe_ctx, btf_path);
	assert(res == 0);
	struct bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);
	struct bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	assert(prog);
	// use the first program and relocate based on btf if btf has been
	// loaded
	res = bpftime_prog_add_helper_group(prog,
					    bpftime_get_ufunc_helper_group());
	assert(res == 0);
	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	return_val = 0;
	res = prog->bpftime_prog_exec(&memory, sizeof(memory), &return_val);
	assert(return_val == 17);
	bpftime_object_close(obj);

	return res;
}
