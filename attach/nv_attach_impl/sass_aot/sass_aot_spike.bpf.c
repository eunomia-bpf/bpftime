// SPDX-License-Identifier: MIT
#include <vmlinux.h>

#ifndef SEC
#define SEC(name) __attribute__((section(name), used))
#endif

SEC("cuda__/sass_aot")
int cuda__sass_aot(struct pt_regs *ctx)
{
	(void)ctx;
	return 42;
}
