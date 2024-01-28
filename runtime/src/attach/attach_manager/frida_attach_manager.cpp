/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach/attach_manager/base_attach_manager.hpp"
#include "frida-gum.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <attach/attach_manager/frida_attach_manager.hpp>
#include <cerrno>
#include <filesystem>
#include <bpftime_ufunc.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <unistd.h>

GType uprobe_listener_get_type();

static std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len != -1) {
		exec_path[len] = '\0'; // Null-terminate the string
		SPDLOG_INFO("Executable path: {}", exec_path);
	} else {
		SPDLOG_ERROR("Error retrieving executable path: {}", errno);
	}
	return exec_path;
}

extern "C" uint64_t __bpftime_frida_attach_manager__replace_handler();
extern "C" void *__bpftime_frida_attach_manager__override_handler();
namespace bpftime
{

frida_attach_manager::~frida_attach_manager()
{
	gum_object_unref(interceptor);
}

frida_attach_manager::frida_attach_manager()
{
	SPDLOG_DEBUG("Initializing frida attach manager");
	gum_init_embedded();
	interceptor = gum_interceptor_obtain();
}

void *frida_attach_manager::resolve_function_addr_by_module_offset(
	const std::string_view &module_name, uintptr_t func_offset)
{
	auto exec_path = get_executable_path();
	void *module_base_addr = nullptr;
	if (std::filesystem::equivalent(module_name, exec_path)) {
		module_base_addr = get_module_base_addr("");
	} else {
		module_base_addr =
			get_module_base_addr(std::string(module_name).c_str());
	}
	if (!module_base_addr) {
		SPDLOG_ERROR("Failed to find module base address for {}",
			     module_name);
		return nullptr;
	}

	return ((char *)module_base_addr) + func_offset;
}

int frida_attach_manager::attach_at(void *func_addr, callback_variant &&cb)
{
	auto itr = internal_attaches.find(func_addr);
	if (itr == internal_attaches.end()) {
		// Create a frida attach entry
		itr = internal_attaches
			      .emplace(func_addr,
				       std::make_unique<
					       frida_internal_attach_entry>(
					       func_addr,
					       (attach_type)cb.index(),
					       interceptor))
			      .first;
		SPDLOG_DEBUG("Created frida attach entry for func addr {:x}",
			     (uintptr_t)func_addr);
	} else if (itr->second->has_override()) {
		SPDLOG_ERROR(
			"Function {} was already attached with replace or filter, cannot attach anything else");
		return -EEXIST;
	}

	auto &inner_attach = itr->second;
	frida_attach_entry ent(next_id, std::move(cb), func_addr);
	int result = ent.self_id;
	next_id++;
	auto inserted_attach_entry =
		this->attaches
			.emplace(ent.self_id,
				 std::make_unique<frida_attach_entry>(
					 std::move(ent)))
			.first;
	inner_attach->user_attaches.push_back(
		inserted_attach_entry->second.get());
	inserted_attach_entry->second->internal_attach = inner_attach.get();
	return result;
}

int frida_attach_manager::attach_uprobe_at(void *func_addr,
					   uprobe_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(
			std::in_place_index_t<(int)attach_type::UPROBE>(), cb));
}

int frida_attach_manager::attach_uretprobe_at(void *func_addr,
					      uretprobe_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(
			std::in_place_index_t<(int)attach_type::URETPROBE>(),
			cb));
}

int frida_attach_manager::attach_uprobe_override_at(
	void *func_addr, uprobe_override_callback &&cb)
{
	return attach_at(
		func_addr,
		callback_variant(std::in_place_index_t<(
					 int)attach_type::UPROBE_OVERRIDE>(),
				 cb));
}

int frida_attach_manager::destroy_attach(int id)
{
	void *drop_func_addr = nullptr;
	if (auto itr = attaches.find(id); itr != attaches.end()) {
		auto p = itr->second->internal_attach;

		auto &user_attaches = p->user_attaches;
		auto tail =
			std::remove_if(user_attaches.begin(),
				       user_attaches.end(),
				       [&](const auto &v) -> bool {
					       return v == itr->second.get();
				       });
		user_attaches.resize(tail - user_attaches.begin());
		attaches.erase(itr);
		if (p->user_attaches.empty()) {
			drop_func_addr = p->function;
		}
	} else {
		SPDLOG_ERROR("Unable to find attach id {}", id);
		errno = -ENOENT;
	}
	if (drop_func_addr)
		internal_attaches.erase(drop_func_addr);
	return 0;
}
void frida_attach_manager::iterate_attaches(attach_iterate_callback cb)
{
	for (const auto &[k, v] : attaches) {
		cb(k, v->function, v->get_type());
	}
}

void *frida_attach_manager::find_function_addr_by_name(const char *name)
{
	if (auto ptr = gum_find_function(name); ptr)
		return ptr;
	if (auto ptr = (void *)gum_module_find_export_by_name(nullptr, name);
	    ptr)
		return ptr;
	return nullptr;
}
int frida_attach_manager::destroy_attach_by_func_addr(const void *func)
{
	if (auto itr = internal_attaches.find((void *)func);
	    itr != internal_attaches.end()) {
		auto uattaches = itr->second->user_attaches;
		for (auto attach_entry : uattaches) {
			attaches.erase(attach_entry->self_id);
		}
		internal_attaches.erase(itr);
		return 0;
	} else {
		return -ENOENT;
	}
}

void *frida_attach_manager::get_module_base_addr(const char *module_name)
{
	gum_module_load(module_name, nullptr);
	return (void *)gum_module_find_base_address(module_name);
}
void *frida_attach_manager::find_module_export_by_name(const char *module_name,
						       const char *symbol_name)
{
	return (void *)(uintptr_t)gum_module_find_export_by_name(module_name,
								 symbol_name);
}

attach_type frida_attach_entry::get_type() const
{
	return (attach_type)cb.index();
}

frida_internal_attach_entry::frida_internal_attach_entry(
	void *function, attach_type basic_attach_type,
	GumInterceptor *interceptor)
	: function(function)
{
	struct interceptor_transaction {
		GumInterceptor *interceptor;
		interceptor_transaction(GumInterceptor *interceptor)
			: interceptor(interceptor)
		{
			gum_interceptor_begin_transaction(interceptor);
		}
		~interceptor_transaction()
		{
			gum_interceptor_end_transaction(interceptor);
		}
	} _transaction(interceptor);
	override_return_callback = nullptr;
	if (basic_attach_type == attach_type::UPROBE ||
	    basic_attach_type == attach_type::URETPROBE) {
		frida_gum_invocation_listener =
			(GumInvocationListener *)g_object_new(
				uprobe_listener_get_type(), NULL);

		if (int err = gum_interceptor_attach(
			    interceptor, function,
			    frida_gum_invocation_listener, this);
		    err < 0) {
			SPDLOG_ERROR(
				"Failed to execute frida gum_interceptor_attach for function {:x}",
				(uintptr_t)function);
			throw std::runtime_error(
				"Failed to attach uprobe/uretprpbe");
		}
	} else if (basic_attach_type == attach_type::UPROBE_OVERRIDE) {
		if (int err = gum_interceptor_replace(
			    interceptor, function,
			    (void *)__bpftime_frida_attach_manager__override_handler,
			    this, nullptr);
		    err < 0) {
			SPDLOG_ERROR(
				"Failed to execute frida replace for function {:x}, when attaching filter, err={}",
				(uintptr_t)function, err);
			throw std::runtime_error("Failed to attach filter");
		}
		override_return_callback = override_return_set_callback(
			[&](uint64_t ctx, uint64_t v) {
				is_overrided = true;
				user_ret = v;
				user_ret_ctx = ctx;
			});
	}
	this->interceptor = gum_object_ref(interceptor);
}

frida_internal_attach_entry::~frida_internal_attach_entry()
{
	SPDLOG_DEBUG("Destroy internal attach at {:x}", (uintptr_t)function);
	if (frida_gum_invocation_listener) {
		gum_interceptor_detach(interceptor,
				       frida_gum_invocation_listener);
		g_object_unref(frida_gum_invocation_listener);
		SPDLOG_DEBUG("Detached listener");
	} else {
		gum_interceptor_revert(interceptor, function);
		SPDLOG_DEBUG("Reverted function replace");
	}
	gum_object_unref(interceptor);
}

bool frida_internal_attach_entry::has_override() const
{
	for (auto v : user_attaches) {
		if (v->get_type() == attach_type::UPROBE_OVERRIDE) {
			return true;
		}
	}
	return false;
}

bool frida_internal_attach_entry::has_uprobe_or_uretprobe() const
{
	for (auto v : user_attaches) {
		if (v->get_type() == attach_type::UPROBE ||
		    v->get_type() == attach_type::URETPROBE) {
			return true;
		}
	}
	return false;
}

base_attach_manager::uprobe_override_callback &
frida_internal_attach_entry::get_filter_callback() const
{
	for (auto v : user_attaches) {
		if (v->get_type() == attach_type::UPROBE_OVERRIDE) {
			return std::get<(int)attach_type::UPROBE_OVERRIDE>(
				v->cb);
		}
	}
	SPDLOG_ERROR(
		"Filter attach not found at function {:x}, but try to get filter callback",
		(uintptr_t)function);
	throw std::runtime_error("Unable to find filter callback");
}

void frida_internal_attach_entry::iterate_uprobe_callbacks(
	const pt_regs &regs) const
{
	for (auto v : user_attaches) {
		if (v->get_type() == attach_type::UPROBE) {
			std::get<(int)attach_type::UPROBE>(v->cb)(regs);
		}
	}
}

void frida_internal_attach_entry::iterate_uretprobe_callbacks(
	const pt_regs &regs) const
{
	for (auto v : user_attaches) {
		if (v->get_type() == attach_type::URETPROBE) {
			std::get<(int)attach_type::URETPROBE>(v->cb)(regs);
		}
	}
}

} // namespace bpftime

using namespace bpftime;

extern "C" void *__bpftime_frida_attach_manager__override_handler()
{
	GumInvocationContext *ctx;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	auto hook_entry = (frida_internal_attach_entry *)
		gum_invocation_context_get_replacement_data(ctx);
	hook_entry->is_overrided = false;
	curr_thread_override_return_callback =
		hook_entry->override_return_callback;

	auto arg0 = gum_invocation_context_get_nth_argument(ctx, 0);
	auto arg1 = gum_invocation_context_get_nth_argument(ctx, 1);
	auto arg2 = gum_invocation_context_get_nth_argument(ctx, 2);
	auto arg3 = gum_invocation_context_get_nth_argument(ctx, 3);
	auto arg4 = gum_invocation_context_get_nth_argument(ctx, 4);
	ufunc_func func = (ufunc_func)ctx->function;

	hook_entry->get_filter_callback()(regs);
	if (hook_entry->is_overrided) {
		return (void *)(uintptr_t)hook_entry->user_ret;
	} else {
		return func((void *)arg0, (void *)arg1, (void *)arg2,
			    (void *)arg3, (void *)arg4);
	}
}

static void uprobe_listener_iface_init(gpointer g_iface, gpointer iface_data);

typedef struct _UprobeListener UprobeListener;

struct _UprobeListener {
	GObject parent;
};

#define EXAMPLE_TYPE_LISTENER (uprobe_listener_get_type())
G_DECLARE_FINAL_TYPE(UprobeListener, uprobe_listener, EXAMPLE, LISTENER,
		     GObject)
G_DEFINE_TYPE_EXTENDED(UprobeListener, uprobe_listener, G_TYPE_OBJECT, 0,
		       G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
					     uprobe_listener_iface_init))

static void uprobe_listener_on_enter(GumInvocationListener *listener,
				     GumInvocationContext *ic)
{
	UprobeListener *self = EXAMPLE_LISTENER(listener);
	auto *hook_entry = (frida_internal_attach_entry *)
		gum_invocation_context_get_listener_function_data(ic);
	SPDLOG_TRACE("Handle uprobe at uprobe_listener_on_enter");
	GumInvocationContext *ctx;
	pt_regs regs;
	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	hook_entry->iterate_uprobe_callbacks(regs);
}

static void uprobe_listener_on_leave(GumInvocationListener *listener,
				     GumInvocationContext *ic)
{
	auto *hook_entry = (frida_internal_attach_entry *)
		gum_invocation_context_get_listener_function_data(ic);
	SPDLOG_TRACE("Handle uretprobe at uprobe_listener_on_leave");
	pt_regs regs;
	GumInvocationContext *ctx;
	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	hook_entry->iterate_uretprobe_callbacks(regs);
}

static void uprobe_listener_class_init(UprobeListenerClass *klass)
{
	(void)EXAMPLE_IS_LISTENER;
}

static void uprobe_listener_iface_init(gpointer g_iface, gpointer iface_data)
{
	GumInvocationListenerInterface *iface =
		(GumInvocationListenerInterface *)g_iface;

	iface->on_enter = uprobe_listener_on_enter;
	iface->on_leave = uprobe_listener_on_leave;
}

static void uprobe_listener_init(UprobeListener *self)
{
}
