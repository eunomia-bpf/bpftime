/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_FRIDA_ATTACH_MANAGER_HPP
#define _BPFTIME_FRIDA_ATTACH_MANAGER_HPP

#include <functional>
#include <variant>
#include <frida_register_def.hpp>
#include <memory>
#include <unordered_map>
#include <base_attach_impl.hpp>
namespace bpftime
{
namespace attach
{
// An attach type that was invoked when the attached function return, receiving
// the return value of that function
constexpr int ATTACH_UPROBE = 6;
// An attach type that was invoked when the attached function was invoked,
// receiving the input args of that function
constexpr int ATTACH_URETPROBE = 7;
// An attach type that was 	run before the original function was invoked.
// Use bpf_override_return to set the return value of the function. If the
// return value is set, the original function will not be invoked.
constexpr int ATTACH_UPROBE_OVERRIDE = 1008;
// Replace the hooked function with the provided callback
constexpr int ATTACH_UREPLACE = 1009;

constexpr int ATTACH_UPROBE_INDEX = 0;
constexpr int ATTACH_URETPROBE_INDEX = 1;
constexpr int ATTACH_UPROBE_OVERRIDE_INDEX = 2;

// Direct callback type for the uprobe attach entries. The argument `regs` will
// be the register state when the function entries
using uprobe_callback = std::function<void(const pt_regs &regs)>;
// Direct callback type for the uretprobe attach entries. The argument `regs`
// will be the register state when the function exits
using uretprobe_callback = std::function<void(const pt_regs &regs)>;
// Direct callback type for the uprobe override attach entries. The argument
// `regs` will be the register state when the function entries. When this was
using uprobe_override_callback = std::function<void(const pt_regs &regs)>;
using callback_variant = std::variant<uprobe_callback, uretprobe_callback,
				      uprobe_override_callback>;
// Callback type used for iterating over all attaches
using attach_iterate_callback =
	std::function<void(int id, const void *addr, int ty)>;

struct ebpf_callback_args {
	ebpf_run_callback ebpf_cb;
	// Used to record the attach type
	int attach_type;
};

// Callback of a attach entry can be either a function that accept pt_regs, or a
// function that accept arguments to the ebpf program
using frida_attach_entry_callback =
	std::variant<callback_variant, ebpf_callback_args>;

// The frida uprobe attach implementation
class frida_attach_impl final : public base_attach_impl {
    public:
	frida_attach_impl();
	~frida_attach_impl();
	// Create a uprobe attach entry at the specified address
	int create_uprobe_at(void *func_addr, uprobe_callback &&cb);
	// Create a uretprobe attach entry at the specified address
	int create_uretprobe_at(void *func_addr, uretprobe_callback &&cb);
	// Create a uprobe override attach entry at the specified address
	int create_uprobe_override_at(void *func_addr,
				      uprobe_override_callback &&cb);
	// Attach at a certain function, with a ebpf function as callback
	int attach_at_with_ebpf_callback(void *func_addr,
					 ebpf_callback_args &&cb);
	// Iterate over all attaches managed by this attach impl instance
	void iterate_attaches(attach_iterate_callback cb);
	// Detach all attach entry at a specified function address
	int detach_by_func_addr(const void *func);

	// Virtual functions
	int detach_by_id(int id);
	// Create an attach entry with ebpf callback
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);
	frida_attach_impl(const frida_attach_impl &) = delete;
	frida_attach_impl &operator=(const frida_attach_impl &) = delete;
	void register_custom_helpers(
		ebpf_helper_register_callback register_callback);

    private:
	void *interceptor;
	std::unordered_map<int, std::unique_ptr<class frida_attach_entry> >
		attaches;
	std::unordered_map<void *,
			   std::unique_ptr<class frida_internal_attach_entry> >
		internal_attaches;

	friend class frida_internal_attach_entry;
	int attach_at(void *func_addr, frida_attach_entry_callback &&cb);
};
} // namespace attach
} // namespace bpftime

#endif
