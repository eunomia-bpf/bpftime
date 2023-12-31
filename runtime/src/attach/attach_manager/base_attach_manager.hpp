/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_ATTACH_MANAGER_HPP
#define _BPFTIME_ATTACH_MANAGER_HPP
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <attach/attach_internal.hpp>
#include <variant>
namespace bpftime
{

using override_return_set_callback = std::function<void(uint64_t, uint64_t)>;
// Used by filter to record the return value
extern thread_local std::optional<override_return_set_callback>
	curr_thread_override_return_callback;

enum class attach_type {
	// Invoked when the attached function return, receiving the return value
	// of that function
	UPROBE = 0,
	// Invoked when the attached function was invoked, receiving the input
	// args of that function
	URETPROBE = 1,
	// Run before the original function was invoked. Use bpf_override_return
	// to set the return value of the function. If the return value is set,
	// the original function will not be invoked.
	UPROBE_OVERRIDE = 2
};

class base_attach_manager {
    public:
	using uprobe_callback = std::function<void(const pt_regs &regs)>;
	using uretprobe_callback = std::function<void(const pt_regs &regs)>;
	using uprobe_override_callback =
		std::function<void(const pt_regs &regs)>;
	using callback_variant =
		std::variant<uprobe_callback, uretprobe_callback,
			     uprobe_override_callback>;
	using attach_iterate_callback =
		std::function<void(int id, const void *addr, attach_type ty)>;

	virtual ~base_attach_manager();
	virtual void *get_module_base_addr(const char *module_name) = 0;
	virtual void *find_module_export_by_name(const char *module_name,
						 const char *symbol_name) = 0;
	virtual void *resolve_function_addr_by_module_offset(
		const std::string_view &module_name, uintptr_t func_offset) = 0;
	virtual int attach_uprobe_at(void *func_addr, uprobe_callback &&cb) = 0;
	virtual int attach_uretprobe_at(void *func_addr,
					uretprobe_callback &&cb) = 0;

	virtual int
	attach_uprobe_override_at(void *func_addr,
				  uprobe_override_callback &&cb) = 0;
	virtual int destroy_attach(int id) = 0;
	virtual int destroy_attach_by_func_addr(const void *func) = 0;
	virtual void iterate_attaches(attach_iterate_callback cb) = 0;
	virtual void *find_function_addr_by_name(const char *name) = 0;
};

} // namespace bpftime

#endif
