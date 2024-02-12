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
using ebpf_run_callback = std::function<int(
	const void *memory, size_t memory_size, uint64_t *return_value)>;

// The base of all attach implementations
class base_attach_impl {
    public:
	// Detach a certain attach entry by a local attach id
	virtual int detach_by_id(int id) = 0;
	// Attach this
	virtual int handle_attach_with_ebpf_call_back(
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
