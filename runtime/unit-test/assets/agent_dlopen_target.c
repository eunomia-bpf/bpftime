/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
__attribute__((noinline)) int bpftime_dlopen_target_func(int x)
{
	volatile int acc = x;
	acc = acc + 7;
	return acc - 6;
}
