#include "base_attach_impl.hpp"
#include <cassert>
#include <spdlog/spdlog.h>

namespace bpftime
{
namespace attach
{
thread_local std::optional<override_return_set_callback>
	curr_thread_override_return_callback;
}
} // namespace bpftime
using namespace bpftime::attach;
int base_attach_impl::allocate_id()
{
	return next_id++;
}

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
		SPDLOG_DEBUG("Overriding return value for ctx {:x} with {}",
			     ctx, value);
		curr_thread_override_return_callback.value()(ctx, value);
	} else {
		SPDLOG_ERROR(
			"Called bpftime_override_return, but no retval callback was set");
		assert(false);
	}
	return 0;
}

base_attach_impl::~base_attach_impl()
{
}
