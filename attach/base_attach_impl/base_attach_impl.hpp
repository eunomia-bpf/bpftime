#ifndef _BPFTIME_BASE_ATTACH_IMPL_HPP
#define _BPFTIME_BASE_ATTACH_IMPL_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include "attach_private_data.hpp"
namespace bpftime
{
namespace attach
{
using override_return_set_callback = std::function<void(uint64_t, uint64_t)>;
// Used by filter to record the return value
extern thread_local std::optional<override_return_set_callback>
	curr_thread_override_return_callback;

// A wrapper function for an entry function of an ebpf program
using ebpf_run_callback = std::function<int(void *memory, size_t memory_size,
					    uint64_t *return_value)>;

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
	int allocate_id();
	virtual ~base_attach_impl();

    private:
	int next_id = 1;
};
} // namespace attach
} // namespace bpftime

#endif
