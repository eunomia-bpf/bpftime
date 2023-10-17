#include "attach/attach_manager/base_attach_manager.hpp"
#include "frida-gum.h"
#include "spdlog/spdlog.h"
#include <attach/attach_manager/frida_attach_manager.hpp>
#include <cerrno>
#include <filesystem>
#include <bpftime_ffi.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
GType uprobe_listener_get_type();

static std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len != -1) {
		exec_path[len] = '\0'; // Null-terminate the string
		spdlog::info("Executable path: {}", exec_path);
	} else {
		spdlog::error("Error retrieving executable path: {}", errno);
	}
	return exec_path;
}

static void *get_module_base_addr(const char *module_name)
{
	gum_module_load(module_name, nullptr);
	return (void *)gum_module_find_base_address(module_name);
}

extern "C" uint64_t __bpftime_frida_attach_manager__replace_handler();
extern "C" void *__bpftime_frida_attach_manager__filter_handler();
namespace bpftime
{
frida_attach_manager::frida_attach_manager()
{
	spdlog::debug("Initializing frida uprobe attach manager");
	gum_init_embedded();
	interceptor = gum_interceptor_obtain();
}

const void *
frida_attach_manager::resolve_function_addr(const std::string_view &module_name,
					    uintptr_t func_offset)
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
		spdlog::error("Failed to find module base address for {}",
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
				       frida_internal_attach_entry(
					       func_addr,
					       (attach_type)cb.index(),
					       interceptor))
			      .first;
	}

	auto &inner_attach = itr->second;
	if (inner_attach->has_replace_or_filter()) {
		spdlog::error(
			"Function {} was already attached with replace or filter, cannot attach uprobe or uretprobe");
		return -EEXIST;
	}
	frida_attach_entry ent(next_id, std::move(cb), func_addr);
	next_id++;
	auto inserted_attach_entry =
		this->attches.emplace(ent.self_id, std::move(ent)).first;
	inner_attach->user_attaches.push_back(inserted_attach_entry->second);
	inserted_attach_entry->second->internal_attaches = inner_attach;
	return 0;
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
int frida_attach_manager::attach_replace_at(void *func_addr,
					    replace_callback &&cb)
{
}
int frida_attach_manager::attach_filter_at(void *func_addr,
					   filter_callback &&cb)
{
}
int frida_attach_manager::destroy_attach(int id)
{
}
void frida_attach_manager::iterate_attaches(attach_iterate_callback cb)
{
	for (const auto &[k, v] : attches) {
		cb(k, v->function, v->get_type());
	}
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
	if (basic_attach_type == attach_type::UPROBE ||
	    basic_attach_type == attach_type::URETPROBE) {
		frida_fum_invocation_listener =
			(GumInvocationListener *)g_object_new(
				uprobe_listener_get_type(), NULL);
		gum_interceptor_begin_transaction(interceptor);
		if (int err = gum_interceptor_attach(
			    interceptor, function,
			    frida_fum_invocation_listener, this);
		    err < 0) {
			spdlog::error(
				"Failed to execute frida gum_interceptor_attach for function {:x}",
				(uintptr_t)function);
			throw std::runtime_error("Failed to attach");
			gum_interceptor_end_transaction(interceptor);
		}
	} else if (basic_attach_type == attach_type::FILTER) {
	} else if (basic_attach_type == attach_type::REPLACE) {
	}
}
bool frida_internal_attach_entry::has_replace_or_filter() const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::REPLACE ||
			    v->get_type() == attach_type::FILTER) {
				return true;
			}
		}
	}
	return false;
}
bool frida_internal_attach_entry::has_uprobe_or_uretprobe() const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::UPROBE ||
			    v->get_type() == attach_type::URETPROBE) {
				return true;
			}
		}
	}
	return false;
}
base_attach_manager::replace_callback &
frida_internal_attach_entry::get_replace_callback() const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::REPLACE) {
				return std::get<
					base_attach_manager::replace_callback>(
					v->cb);
			}
		}
	}
	spdlog::error(
		"Replace attach not found at function {:x}, but try to get replace callback",
		(uintptr_t)function);
	throw std::runtime_error("Unable to find replace callback");
}
base_attach_manager::filter_callback &
frida_internal_attach_entry::get_filter_callback() const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::FILTER) {
				return std::get<
					base_attach_manager::filter_callback>(
					v->cb);
			}
		}
	}
	spdlog::error(
		"Filter attach not found at function {:x}, but try to get filter callback",
		(uintptr_t)function);
	throw std::runtime_error("Unable to find filter callback");
}
void frida_internal_attach_entry::iterate_uprobe_callbacks(
	const pt_regs &regs) const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::UPROBE) {
				std::get<(int)attach_type::UPROBE>(v->cb)(regs);
			}
		}
	}
}
void frida_internal_attach_entry::iterate_uretprobe_callbacks(
	const pt_regs &regs) const
{
	for (auto p : user_attaches) {
		if (auto v = p.lock(); v) {
			if (v->get_type() == attach_type::URETPROBE) {
				std::get<(int)attach_type::URETPROBE>(v->cb)(
					regs);
			}
		}
	}
}

} // namespace bpftime

using namespace bpftime;

extern "C" uint64_t __bpftime_frida_attach_manager__replace_handler()
{
	GumInvocationContext *ctx;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	frida_internal_attach_entry *hook_entry =
		(frida_internal_attach_entry *)
			gum_invocation_context_get_replacement_data(ctx);
	auto &cb = hook_entry->get_replace_callback();
	uint64_t ret = cb(regs);
	gum_invocation_context_replace_return_value(ctx, (gpointer)ret);
	return ret;
}
extern "C" void *__bpftime_frida_attach_manager__filter_handler()
{
	GumInvocationContext *ctx;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	auto hook_entry = (frida_internal_attach_entry *)
		gum_invocation_context_get_replacement_data(ctx);
	uint64_t user_ret = 0;
	curr_thread_set_ret_val =
		retval_set_callback([&](uint64_t v) { user_ret = v; });

	auto arg0 = gum_invocation_context_get_nth_argument(ctx, 0);
	auto arg1 = gum_invocation_context_get_nth_argument(ctx, 1);
	auto arg2 = gum_invocation_context_get_nth_argument(ctx, 2);
	auto arg3 = gum_invocation_context_get_nth_argument(ctx, 3);
	auto arg4 = gum_invocation_context_get_nth_argument(ctx, 4);
	ffi_func func = (ffi_func)ctx->function;

	auto skip = hook_entry->get_filter_callback()(regs);
	if (skip) {
		return (void *)(uintptr_t)user_ret;
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
	spdlog::trace("Handle uprobe at uprobe_listener_on_enter");
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
	spdlog::trace("Handle uretprobe at uprobe_listener_on_leave");
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
