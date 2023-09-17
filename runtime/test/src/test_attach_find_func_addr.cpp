#include "bpftime.hpp"
#include "test_defs.h"
#include <stdio.h>
#include <cstring>
#include "bpftime_shm.hpp"

using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_NO_CREATE;

extern "C" void test_with_builtin_elf(bpf_attach_ctx *probe_ctx,
				  const char *symbol_name, void *symbol_ptr)
{
	char *err_msg = NULL;
	void *func_addr;
	func_addr = probe_ctx->find_function_by_name(symbol_name);
	printf("func_addr %p\n", func_addr);
	printf("%s addr %p\n", symbol_name, symbol_ptr);
	if (err_msg) {
		printf("err %s\n", err_msg);
		free(err_msg);
		return;
	}
	assert(func_addr != NULL);
	assert(func_addr == symbol_ptr);
}

struct test_cases {
	const char *symbol_name;
	void *symbol_ptr;
};

int main()
{
	// test generate offset
	system(generate_nm_offset_command("test_probe_find_func_addr"));
	char *err_msg = NULL;
	bpf_attach_ctx probe_ctx;

	struct test_cases test_cases[] = {
		{ "main", (void*)main },
		{ "test_with_builtin_elf", (void*)test_with_builtin_elf },
		{ "strlen", (void*)strlen }
	};

	for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
		test_with_builtin_elf(&probe_ctx, test_cases[i].symbol_name,
				      test_cases[i].symbol_ptr);
	}
	auto module_base_libc = probe_ctx.module_get_base_addr("libc.so.6");
	auto module_base_self = probe_ctx.module_get_base_addr("");
	printf("libc base %p, empty base %p\n", module_base_libc, module_base_self);
	assert(module_base_libc != NULL && module_base_self != NULL);
	return 0;
}
