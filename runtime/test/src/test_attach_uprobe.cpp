#include <stdio.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <assert.h>
#include <unistd.h>
#include <inttypes.h>
#include "bpftime.hpp"
#include "bpftime_object.h"
#include "test_defs.h"
#include "bpftime_shm.hpp"

using namespace bpftime;
const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_NO_CREATE;

// This is the original function to hook.
int my_function(int parm1, const char *str, char c)
{
	printf("origin func: Args: %d, %s, %c\n", parm1, str, c);
	return 35;
}

const char *obj_path = "./uprobe.bpf.o";

bpftime_prog *get_prog(const char *name, bpftime_object *obj)
{
	bpftime_prog *prog = bpftime_object_find_program_by_name(obj, name);
	assert(prog);
	// add ffi support
	int res = bpftime_helper_group::get_kernel_utils_helper_group()
			  .add_helper_group_to_prog(prog);
	assert(res == 0);
	res = prog->bpftime_prog_load(false);
	assert(res == 0);
	return prog;
}

void test_attach_to_local_function(bpf_attach_ctx &probe_ctx,
				   bpftime_object *obj)
{
	// get the first program
	auto my_function_uprobe_prog = get_prog("my_function_uprobe", obj);
	// attach
	int fd = probe_ctx.create_uprobe((void *)my_function, 1);
	assert(fd >= 0);
	int res = probe_ctx.attach_prog(my_function_uprobe_prog, fd);
	assert(res == 0);
	// attach again
	res = probe_ctx.attach_prog(my_function_uprobe_prog, fd);
	assert(res == 0);

	// test attach uretprobe
	int ret_fd = probe_ctx.create_uprobe((void *)my_function, 2, true);
	assert(ret_fd >= 0);
	auto my_function_uretprobe_prog =
		get_prog("my_function_uretprobe", obj);
	res = probe_ctx.attach_prog(my_function_uretprobe_prog, ret_fd);
	assert(res == 0);

	// test for attach
	res = my_function(1, "hello aaa", 'c');
	printf("hooked func return: %d\n", res);

	res = probe_ctx.destory_attach(fd);
	assert(res == 0);
	res = probe_ctx.destory_attach(ret_fd);
	assert(res == 0);
}

void test_attach_to_libc_function(bpf_attach_ctx &probe_ctx,
				  bpftime_object *obj)
{
	auto module_base_libc = probe_ctx.module_get_base_addr("libc.so.6");
	auto strdup_addr = probe_ctx.module_find_export_by_name(NULL, "strdup");
	auto strdup_addr_libc =
		probe_ctx.module_find_export_by_name("libc.so.6", "strdup");
	assert(strdup_addr_libc == strdup_addr);
	printf("libc base %p, strlen addr %p， offset %lu\n", module_base_libc,
	       strdup_addr, (uint64_t)strdup_addr - (uint64_t)module_base_libc);
	// get the first program
	auto strdup_uprobe_prog = get_prog("strdup_uprobe", obj);
	// attach
	int fd = probe_ctx.create_uprobe((void *)strdup_addr, 3);
	assert(fd >= 0);
	int res = probe_ctx.attach_prog(strdup_uprobe_prog, fd);
	assert(res == 0);

	// test attach uretprobe
	int ret_fd = probe_ctx.create_uprobe((void *)strdup_addr, 4, true);
	assert(ret_fd >= 0);
	auto strdup_uretprobe_prog = get_prog("strdup_uretprobe", obj);
	res = probe_ctx.attach_prog(strdup_uretprobe_prog, ret_fd);
	assert(res == 0);

	// test for attach
	char *dup = strdup("hello aaa");

	printf("hooked func return: %d\n", res);

	res = probe_ctx.destory_attach(fd);
	assert(res == 0);
	res = probe_ctx.destory_attach(ret_fd);
	assert(res == 0);
	free(dup);
}

int main()
{
	int res = 1;

	// test for no attach
	res = my_function(1, "hello aaa", 'c');
	printf("origin func return: %d\n", res);
	assert(res == 35);

	puts("test attach");

	bpf_attach_ctx probe_ctx;
	bpftime_object *obj = bpftime_object_open(obj_path);
	assert(obj);

	test_attach_to_local_function(probe_ctx, obj);
	test_attach_to_libc_function(probe_ctx, obj);
	bpftime_object_close(obj);

	return 0;
}
