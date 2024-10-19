#ifndef _BPFTIME_BASE_ATTACH_IMPL_HPP
#define _BPFTIME_BASE_ATTACH_IMPL_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include "attach_private_data.hpp"
#include <stdexcept>
namespace bpftime
{
namespace attach
{
using override_return_set_callback = std::function<void(uint64_t, uint64_t)>;

// Used by filter to record the return value
// Use inline thread_local to ensure ODR
inline thread_local std::optional<override_return_set_callback>
	curr_thread_override_return_callback;

// A wrapper function for an entry function of an ebpf program
using ebpf_run_callback = std::function<int(void *memory, size_t memory_size,
					    uint64_t *return_value)>;

// A callback to register bpf helpers
using ebpf_helper_register_callback =
	std::function<int(unsigned index, const char *name, void *func)>;
// The base of all attach implementations
class base_attach_impl {
    public:
	// Detach a certain attach entry by a local attach id
	virtual int detach_by_id(int id) = 0;

	// Create an attach entry with an ebpf callback. To avoid messing up the
	// code base, we don't use bpftime_prog here, instead, we require a
	// callback accept the same argument as bpftime_prog::run The callback
	// would be called if the attach entry is triggered. The specific attach
	// impl will be responsible for preparing arguments to the ebpf program.
	virtual int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type) = 0;

	// Allocate a new attach entry id
	int allocate_id()
	{
		return next_id++;
	}

	// Attach manager will call this function, so that attach impls could
	// register their custom helpers
	virtual void
	register_custom_helpers(ebpf_helper_register_callback register_callback)
	{
	}

	virtual ~base_attach_impl(){};

    private:
	int next_id = 1;
};
} // namespace attach
} // namespace bpftime

// The use of extern "C" allows the function to be called from C code
extern "C" {

// Set the return value of the current context
inline uint64_t bpftime_set_retval(uint64_t value)
{
	using namespace bpftime::attach;
	if (curr_thread_override_return_callback.has_value()) {
		curr_thread_override_return_callback.value()(0, value);
	} else {
		spdlog::error(
			"Called bpftime_set_retval, but no retval callback was set");
		throw std::invalid_argument(
			"Called bpftime_set_retval, but no retval callback was set");
	}
	return 0;
}

// Override the return value of the current context
inline uint64_t bpftime_override_return(uint64_t ctx, uint64_t value)
{
	using namespace bpftime::attach;
	if (curr_thread_override_return_callback.has_value()) {
		spdlog::debug("Overriding return value for ctx {:x} with {}",
			      ctx, value);
		curr_thread_override_return_callback.value()(ctx, value);
	} else {
		spdlog::error(
			"Called bpftime_override_return, but no retval callback was set");
		throw std::invalid_argument(
			"Called bpftime_override_return, but no retval callback was set");
	}
	return 0;
}

} // extern "C"

#endif
