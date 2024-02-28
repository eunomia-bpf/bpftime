#ifndef _BPFTIME_FRIDA_INTERNAL_ATTACH_ENTRY_HPP
#define _BPFTIME_FRIDA_INTERNAL_ATTACH_ENTRY_HPP
#include "frida_attach_entry.hpp"
#include <frida-gum.h>
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
	GumInvocationListener *frida_gum_invocation_listener = nullptr;

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
	void run_filter_callback(const pt_regs &regs) const;
	void iterate_uprobe_callbacks(const pt_regs &regs) const;
	void iterate_uretprobe_callbacks(const pt_regs &regs) const;
	frida_internal_attach_entry(const frida_internal_attach_entry &) =
		delete;
	frida_internal_attach_entry &
	operator=(const frida_internal_attach_entry &) = delete;
	~frida_internal_attach_entry();
	frida_internal_attach_entry(void *function, int basic_attach_type,
				    GumInterceptor *interceptor);
	frida_internal_attach_entry(frida_internal_attach_entry &&) = default;
};
} // namespace attach
} // namespace bpftime
#endif
