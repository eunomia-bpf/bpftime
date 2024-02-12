/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_FRIDA_ATTACH_MANAGER_HPP
#define _BPFTIME_FRIDA_ATTACH_MANAGER_HPP

#include <functional>
#include <variant>
#include "frida_uprobe_attach_internal.hpp"
#include <memory>
#include <unordered_map>
#include <base_attach_impl.hpp>
#include <frida-gum.h>
namespace bpftime
{
namespace attach
{

constexpr int ATTACH_UPROBE = 0;
constexpr int ATTACH_URETPROBE = 1;
constexpr int ATTACH_UPROBE_OVERRIDE = 2;

using uprobe_callback = std::function<void(const pt_regs &regs)>;
using uretprobe_callback = std::function<void(const pt_regs &regs)>;
using uprobe_override_callback = std::function<void(const pt_regs &regs)>;
using callback_variant = std::variant<uprobe_callback, uretprobe_callback,
				      uprobe_override_callback>;
using attach_iterate_callback =
	std::function<void(int id, const void *addr, int ty)>;

class frida_attach_manager final : public base_attach_impl {
    public:
	frida_attach_manager();
	~frida_attach_manager();

	int attach_uprobe_at(void *func_addr, uprobe_callback &&cb);
	int attach_uretprobe_at(void *func_addr, uretprobe_callback &&cb);
	int attach_uprobe_override_at(void *func_addr,
				      uprobe_override_callback &&cb);
	void iterate_attaches(attach_iterate_callback cb);
	int destroy_attach_by_func_addr(const void *func);

	// Virtual functions
	int detach_by_id(int id);
	int handle_attach_with_ebpf_call_back(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type);

    private:
	// Use void pointer to avoid leaking frida-gum.h to other parts
	GumInterceptor *interceptor;
	std::unordered_map<int, std::unique_ptr<class frida_attach_entry> >
		attaches;
	std::unordered_map<void *,
			   std::unique_ptr<class frida_internal_attach_entry> >
		internal_attaches;

	friend class frida_internal_attach_entry;
	int attach_at(void *func_addr, callback_variant &&cb);
};
} // namespace attach
} // namespace bpftime

#endif
