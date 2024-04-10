#include "frida_internal_attach_entry.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "frida_attach_entry.hpp"
#include <spdlog/spdlog.h>
#include <frida_register_conversion.hpp>
using namespace bpftime::attach;
GType uprobe_listener_get_type();

extern "C" uint64_t __bpftime_frida_attach_manager__replace_handler();
extern "C" void *__bpftime_frida_attach_manager__override_handler();

frida_internal_attach_entry::frida_internal_attach_entry(
	void *function, int basic_attach_type, GumInterceptor *interceptor)
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
	if (basic_attach_type == ATTACH_UPROBE ||
	    basic_attach_type == ATTACH_URETPROBE) {
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
	} else if (basic_attach_type == ATTACH_UPROBE_OVERRIDE) {
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
				SPDLOG_DEBUG(
					"Frida attach manager: received override return, value {}, context {:x}",
					v, ctx);
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
	SPDLOG_DEBUG("Destructor of frida_internal_attach_entry exiting..");
}

bool frida_internal_attach_entry::has_override() const
{
	for (auto v : user_attaches) {
		if (v->get_type() == ATTACH_UPROBE_OVERRIDE) {
			return true;
		}
	}
	return false;
}

bool frida_internal_attach_entry::has_uprobe_or_uretprobe() const
{
	for (auto v : user_attaches) {
		if (v->get_type() == ATTACH_UPROBE ||
		    v->get_type() == ATTACH_URETPROBE) {
			return true;
		}
	}
	return false;
}

void frida_internal_attach_entry::run_filter_callback(const pt_regs &regs) const
{
	for (auto v : user_attaches) {
		if (v->get_type() == ATTACH_UPROBE_OVERRIDE) {
			v->run_callback<ATTACH_UPROBE_OVERRIDE_INDEX>(regs);
			// There should be at most one filter attach..
			return;
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
		if (v->get_type() == ATTACH_UPROBE) {
			v->run_callback<ATTACH_UPROBE_INDEX>(regs);
		}
	}
}

void frida_internal_attach_entry::iterate_uretprobe_callbacks(
	const pt_regs &regs) const
{
	for (auto v : user_attaches) {
		if (v->get_type() == ATTACH_URETPROBE) {
			v->run_callback<ATTACH_URETPROBE_INDEX>(regs);
		}
	}
}

typedef void *(*ufunc_func)(void *r1, void *r2, void *r3, void *r4, void *r5);

extern "C" void *__bpftime_frida_attach_manager__override_handler()
{
	GumInvocationContext *ctx;
	bpftime::pt_regs regs;

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

	hook_entry->run_filter_callback(regs);
	if (hook_entry->is_overrided) {
		auto value = (uintptr_t)hook_entry->user_ret;
		SPDLOG_DEBUG("Using override return value: {}", value);
		return (void *)value;
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
	bpftime::pt_regs regs;
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
	bpftime::pt_regs regs;
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
