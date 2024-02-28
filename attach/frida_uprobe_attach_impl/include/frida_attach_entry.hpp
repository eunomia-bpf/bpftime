#ifndef _BPFTIME_FRIDA_ATTACH_ENTRY_HPP
#define _BPFTIME_FRIDA_ATTACH_ENTRY_HPP

#include "base_attach_impl.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <variant>
namespace bpftime
{
namespace attach
{
// Used to represent an attach entry faced to users. It's related with a unique
// id, and holds a callback provided by the user
class frida_attach_entry {
	int self_id;
	frida_attach_entry_callback callback;

	void *function;

	class frida_internal_attach_entry *internal_attach;
	friend class frida_attach_impl;
	friend class frida_internal_attach_entry;

	template <int callback_index>
	void run_callback(const pt_regs &regs) const
	{
		if (std::holds_alternative<callback_variant>(callback)) {
			SPDLOG_DEBUG(
				"Run filter callback with original callback");
			std::get<callback_index>(
				std::get<callback_variant>(callback))(regs);
		} else {
			auto &ebpf_call_args =
				std::get<ebpf_callback_args>(callback);

			SPDLOG_DEBUG(
				"Run filter callback with ebpf callback function, type {}",
				ebpf_call_args.attach_type);
			uint64_t ret = 0;
			int err = ebpf_call_args.ebpf_cb((void *)&regs,
							 sizeof(regs), &ret);
			if (err < 0) {
				SPDLOG_ERROR("Unable to run ebpf callback: {}",
					     err);
			}
		}
	}

    public:
	// Get the specific attach type of this attach entry
	// Could be ATTACH_UPROBE, ATTACH_URETPROBE, ATTACH_UPROBE_OVERRIDE
	int get_type() const;
	frida_attach_entry(const frida_attach_entry &) = delete;
	frida_attach_entry &operator=(const frida_attach_entry &) = delete;
	// Create this attach entry with its id, callback function, and function
	// address to hook
	frida_attach_entry(int id, frida_attach_entry_callback &&cb, void *function)
		: self_id(id), callback(cb), function(function)
	{
	}
	frida_attach_entry(frida_attach_entry &&) = default;
};
} // namespace attach
} // namespace bpftime

#endif
