#include <stdio.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <inttypes.h>
#include <assert.h>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"

using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type =
	shm_open_type::SHM_NO_CREATE;

// This is the hook function.
int my_hook_function()
{
	printf("Hello from hook!\n");
	return 11;
}

// This is the original function to hook.
int my_function()
{
	printf("Hello, world!\n");
	return 67;
}

unsigned char orig_bytes[256];

int main()
{
	int res [[maybe_unused]] = my_function();
	assert(res == 67);

	bpf_attach_ctx probe_ctx;

	probe_ctx.replace_func((void *)my_hook_function, (void *)my_function,
			       NULL);

	// Now calling the function will actually call the hook function.
	res = my_function();
	assert(res == 11);

	probe_ctx.revert_func((void *)my_function);

	// Now calling the function will call the original function.
	res = my_function();
	assert(res == 67);

	return 0;
}
