#ifndef _BPFTIME_FRIDA_ATTACH_MANAGER_HPP
#define _BPFTIME_FRIDA_ATTACH_MANAGER_HPP

#include "bpf_attach_ctx.hpp"
#include "frida-gum.h"
#include <attach/attach_manager/base_attach_manager.hpp>
#include <memory>
#include <unordered_map>

namespace bpftime
{

// Used to represent an attach entry faced to users. It's related with a unique
// id, and holds a callback provided by the user
class frida_attach_entry {
	int self_id;
	base_attach_manager::callback_variant cb;
	void *function;

	std::weak_ptr<class frida_internal_attach_entry> internal_attaches;
	friend class frida_attach_manager;
	friend class frida_internal_attach_entry;

    public:
	attach_type get_type() const;
	frida_attach_entry(const frida_attach_entry &) = delete;
	frida_attach_entry &operator=(const frida_attach_entry &) = delete;
	frida_attach_entry(int id, base_attach_manager::callback_variant &&cb,
			   void *function)
		: self_id(id), cb(cb), function(function)
	{
	}
	frida_attach_entry(frida_attach_entry &&) = default;
};
// Used to represent an attach entry faced to frida. On a certain function, we
// can have either one replace attach, or one filter attach, or many uprobe
// attaches or uretprobe attaches. This entry was related with a function
// address
class frida_internal_attach_entry {
	void *function;
	GumInterceptor *interceptor;
	std::vector<std::weak_ptr<frida_attach_entry> > user_attaches;
	GumInvocationListener *frida_gum_invocation_listener = nullptr;

	friend class frida_attach_manager;

    public:
	bool has_replace_or_filter() const;
	bool has_uprobe_or_uretprobe() const;
	base_attach_manager::replace_callback &get_replace_callback() const;
	base_attach_manager::filter_callback &get_filter_callback() const;
	void iterate_uprobe_callbacks(const pt_regs &regs) const;
	void iterate_uretprobe_callbacks(const pt_regs &regs) const;
	frida_internal_attach_entry(const frida_internal_attach_entry &) =
		delete;
	frida_internal_attach_entry &
	operator=(const frida_internal_attach_entry &) = delete;
	~frida_internal_attach_entry();
	frida_internal_attach_entry(void *function,
				    attach_type basic_attach_type,
				    GumInterceptor *interceptor);
	frida_internal_attach_entry(frida_internal_attach_entry &&) = default;
};

class frida_attach_manager final : public base_attach_manager {
    public:
	frida_attach_manager();
	~frida_attach_manager();

	void *resolve_function_addr_by_module_offset(
		const std::string_view &module_name, uintptr_t func_offset);
	int attach_uprobe_at(void *func_addr, uprobe_callback &&cb);
	int attach_uretprobe_at(void *func_addr, uretprobe_callback &&cb);
	int attach_replace_at(void *func_addr, replace_callback &&cb);
	int attach_filter_at(void *func_addr, filter_callback &&cb);
	int destroy_attach(int id);
	void iterate_attaches(attach_iterate_callback cb);

	void *find_function_addr_by_name(const char *name);

	int destroy_attach_by_func_addr(const void *func);
	void *get_module_base_addr(const char *module_name);
	void *find_module_export_by_name(const char *module_name,
					 const char *symbol_name);

    private:
	GumInterceptor *interceptor;
	int next_id = 1;
	std::unordered_map<int, std::shared_ptr<frida_attach_entry> > attaches;
	std::unordered_map<void *, std::shared_ptr<frida_internal_attach_entry> >
		internal_attaches;

	friend class frida_internal_attach_entry;
	int attach_at(void *func_addr, callback_variant &&cb);
};
} // namespace bpftime

#endif
