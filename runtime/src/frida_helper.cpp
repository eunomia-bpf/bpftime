/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <cstdint>
#include <frida-gum.h>

extern "C" uint64_t bpftime_get_func_ip(uint64_t, uint64_t, uint64_t, uint64_t,
					uint64_t)
{
	GumInvocationContext *ctx = gum_interceptor_get_current_invocation();
	return ctx == nullptr ? 0 : reinterpret_cast<uintptr_t>(ctx->function);
}
