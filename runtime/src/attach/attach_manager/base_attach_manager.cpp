#include "spdlog/spdlog.h"
#include <attach/attach_manager/base_attach_manager.hpp>
#include <optional>

namespace bpftime
{
thread_local std::optional<retval_set_callback> curr_thread_set_ret_val;

extern "C" uint64_t bpftime_get_retval(void)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	return (uintptr_t)gum_invocation_context_get_return_value(gum_ctx);
}

extern "C" uint64_t bpftime_set_retval(uint64_t value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}

	if (curr_thread_set_ret_val.has_value()) {
		curr_thread_set_ret_val.value()(value);
	} else {
		spdlog::error(
			"Called bpftime_set_retval, but no retval callback was set");
	}
	gum_invocation_context_replace_return_value(gum_ctx,
						    (gpointer)((size_t)value));
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

} // namespace bpftime
