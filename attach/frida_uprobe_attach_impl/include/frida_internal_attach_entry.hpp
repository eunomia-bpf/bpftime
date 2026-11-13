#ifndef _BPFTIME_FRIDA_INTERNAL_ATTACH_ENTRY_HPP
#define _BPFTIME_FRIDA_INTERNAL_ATTACH_ENTRY_HPP
#include "frida_attach_entry.hpp"
#include <frida-gum.h>
#include <atomic>
#include <vector>
#include "frida_uprobe_attach_impl.hpp"
namespace bpftime
{
namespace attach
{
// Used to represent an attach entry faced to frida. On a certain function, we
// can have either one replace attach, or one filter attach, or many uprobe
// attaches or uretprobe attaches. This entry was related with a function
// address
class frida_internal_attach_entry {
	void *function;
	GumInterceptor *interceptor;
	std::vector<frida_attach_entry *> user_attaches;
	GumInvocationListener *uprobe_listener = nullptr;
	GumInvocationListener *uretprobe_listener = nullptr;
	void *short_uprobe_stub = nullptr;
	uint64_t short_uprobe_original = 0;
	// Number of threads currently executing the short-uprobe stub. The stub
	// page and this entry must not be reclaimed while it is non-zero.
	mutable std::atomic<unsigned> active_short_callbacks{ 0 };

	friend class frida_attach_impl;

    public:
	// Whether the function `bpftime_set_retval` is invoked. If this is set
	// to true, the return value should be overrided by `user_ret`
	bool is_overrided = false;
	uint64_t user_ret = 0;
	// Extra context when overriding the return value
	uint64_t user_ret_ctx = 0;
	override_return_set_callback override_return_callback;

	bool has_override() const;
	bool has_uprobe_or_uretprobe() const;
	void ensure_listener(int attach_type);
	void run_filter_callback(const pt_regs &regs) const;
	void iterate_uprobe_callbacks(const pt_regs &regs) const;
	void iterate_uretprobe_callbacks(const pt_regs &regs) const;
	void *get_function_address() const { return function; }

	// Tracks one in-flight short-uprobe callback so the destructor can wait
	// for all threads to leave the stub before its page and this entry are
	// reclaimed.
	struct short_callback_guard {
		frida_internal_attach_entry &entry;
		explicit short_callback_guard(
			frida_internal_attach_entry &entry)
			: entry(entry)
		{
			entry.active_short_callbacks.fetch_add(
				1, std::memory_order_acq_rel);
		}
		~short_callback_guard()
		{
			entry.active_short_callbacks.fetch_sub(
				1, std::memory_order_acq_rel);
		}
		short_callback_guard(const short_callback_guard &) = delete;
		short_callback_guard &
		operator=(const short_callback_guard &) = delete;
	};
	frida_internal_attach_entry(const frida_internal_attach_entry &) =
		delete;
	frida_internal_attach_entry &
	operator=(const frida_internal_attach_entry &) = delete;
	~frida_internal_attach_entry();
	frida_internal_attach_entry(void *function, int basic_attach_type,
				    GumInterceptor *interceptor);
	// Entries hold an atomic in-flight counter and raw stub resources and are
	// only ever held in place via unique_ptr, so moving is not supported.
	frida_internal_attach_entry(frida_internal_attach_entry &&) = delete;
};
} // namespace attach
} // namespace bpftime
#endif
