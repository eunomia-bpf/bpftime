/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <attach/attach_manager/base_attach_manager.hpp>
#include <optional>

namespace bpftime
{
thread_local std::optional<override_return_set_callback> curr_thread_override_return_callback;

extern "C" uint64_t bpftime_get_retval(void)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	return (uintptr_t)gum_invocation_context_get_return_value(gum_ctx);
}

base_attach_manager::~base_attach_manager()
{
}
} // namespace bpftime

extern "C" uint64_t bpftime_set_retval(uint64_t value)
{
	using namespace bpftime;
	if (curr_thread_override_return_callback.has_value()) {
		curr_thread_override_return_callback.value()(0, value);
	} else {
		SPDLOG_ERROR(
			"Called bpftime_set_retval, but no retval callback was set");
		assert(false);
	}
	return 0;
}

extern "C" uint64_t bpftime_override_return(uint64_t ctx, uint64_t value)
{
	using namespace bpftime;
	if (curr_thread_override_return_callback.has_value()) {
		curr_thread_override_return_callback.value()(ctx, value);
	} else {
		SPDLOG_ERROR(
			"Called bpftime_set_retval, but no retval callback was set");
		assert(false);
	}
	return 0;
}


extern "C" uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	// ignore ctx;
	*value = (uint64_t)gum_invocation_context_get_return_value(gum_ctx);
	return 0;
}

extern "C" uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n,
					 uint64_t *value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EINVAL;
	}
	// ignore ctx;
	*value = (uint64_t)gum_cpu_context_get_nth_argument(
		gum_ctx->cpu_context, n);
	return 0;
}
