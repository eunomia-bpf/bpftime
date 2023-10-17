#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include "bpftime.hpp"
#include "bpftime_internal.h"
#include <frida-gum.h>
#include <syscall_table.hpp>
#include <spdlog/spdlog.h>
#include <attach/attach_internal.hpp>
using namespace std;
using namespace bpftime;

namespace bpftime
{

// uint64_t __frida_bpftime_replace_handler()
// {
// 	GumInvocationContext *ctx;
// 	struct hook_entry *hook_entry;
// 	pt_regs regs;

// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	hook_entry = (struct hook_entry *)
// 		gum_invocation_context_get_replacement_data(ctx);
// 	auto ptr = hook_entry->progs.begin();
// 	if (ptr == hook_entry->progs.end()) {
// 		return 0;
// 	}
// 	const bpftime_prog *prog = *ptr;
// 	if (!prog) {
// 		return 0;
// 	}
// 	uint64_t ret_val;
// 	int res = prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
// 	if (res < 0) {
// 		return 0;
// 	}
// 	gum_invocation_context_replace_return_value(ctx, (gpointer)ret_val);
// 	return ret_val;
// }

// void *__frida_bpftime_filter_handler()
// {
// 	GumInvocationContext *ctx;
// 	struct hook_entry *hook_entry;
// 	pt_regs regs;

// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	hook_entry = (struct hook_entry *)
// 		gum_invocation_context_get_replacement_data(ctx);
// 	auto ptr = hook_entry->progs.begin();
// 	if (ptr == hook_entry->progs.end()) {
// 		return 0;
// 	}
// 	const bpftime_prog *prog = *ptr;
// 	if (!prog) {
// 		return 0;
// 	}
// 	// naive implementation
// 	// TODO: is there any better ways?
// 	auto arg0 = gum_invocation_context_get_nth_argument(ctx, 0);
// 	auto arg1 = gum_invocation_context_get_nth_argument(ctx, 1);
// 	auto arg2 = gum_invocation_context_get_nth_argument(ctx, 2);
// 	auto arg3 = gum_invocation_context_get_nth_argument(ctx, 3);
// 	auto arg4 = gum_invocation_context_get_nth_argument(ctx, 4);
// 	ffi_func func = (ffi_func)ctx->function;
// 	uint64_t op_code;
// 	int res = prog->bpftime_prog_exec(&regs, sizeof(regs), &op_code);
// 	if (res < 0) {
// 		return 0;
// 	}
// 	// printf("filter op code: %" PRId64 " ret: %" PRIuPTR "\n",
// 	// op_code,
// 	//        (uintptr_t)hook_entry->ret_val);
// 	if (op_code == OP_SKIP) {
// 		// drop the function call and not proceed
// 		return hook_entry->ret_val;
// 	} else if (op_code != OP_RESUME) {
// 		g_printerr("Invalid op code %" PRId64 "\n", op_code);
// 		return (void *)(uintptr_t)op_code;
// 	}

// 	// recover the origin function
// 	return func((void *)arg0, (void *)arg1, (void *)arg2, (void *)arg3,
// 		    (void *)arg4);
// }

// typedef struct _UprobeListener UprobeListener;

// struct _UprobeListener {
// 	GObject parent;
// };

// static void uprobe_listener_iface_init(gpointer g_iface, gpointer iface_data);

// #define EXAMPLE_TYPE_LISTENER (uprobe_listener_get_type())
// G_DECLARE_FINAL_TYPE(UprobeListener, uprobe_listener, EXAMPLE, LISTENER,
// 		     GObject)
// G_DEFINE_TYPE_EXTENDED(UprobeListener, uprobe_listener, G_TYPE_OBJECT, 0,
// 		       G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
// 					     uprobe_listener_iface_init))

// static void uprobe_listener_on_enter(GumInvocationListener *listener,
// 				     GumInvocationContext *ic)
// {
// 	UprobeListener *self = EXAMPLE_LISTENER(listener);
// 	hook_entry *hook_entry = (struct hook_entry *)
// 		gum_invocation_context_get_listener_function_data(ic);
// 	if (hook_entry->progs.size() == 0) {
// 		return;
// 	}
// 	spdlog::trace("Handle uprobe at uprobe_listener_on_enter");
// 	GumInvocationContext *ctx;
// 	pt_regs regs;
// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	for (auto &prog : hook_entry->progs) {
// 		uint64_t ret_val;
// 		int res =
// 			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
// 		if (res < 0) {
// 			return;
// 		}
// 	}
// }

// static void uprobe_listener_on_leave(GumInvocationListener *listener,
// 				     GumInvocationContext *ic)
// {
// 	hook_entry *hook_entry = (struct hook_entry *)
// 		gum_invocation_context_get_listener_function_data(ic);
// 	if (hook_entry->ret_progs.size() == 0) {
// 		return;
// 	}
// 	spdlog::trace("Handle uretprobe at uprobe_listener_on_leave");
// 	pt_regs regs;
// 	GumInvocationContext *ctx;
// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	for (auto &prog : hook_entry->ret_progs) {
// 		uint64_t ret_val;
// 		int res =
// 			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
// 		if (res < 0) {
// 			return;
// 		}
// 	}
// }

// static void uprobe_listener_class_init(UprobeListenerClass *klass)
// {
// 	(void)EXAMPLE_IS_LISTENER;
// }

// static void uprobe_listener_iface_init(gpointer g_iface, gpointer iface_data)
// {
// 	GumInvocationListenerInterface *iface =
// 		(GumInvocationListenerInterface *)g_iface;

// 	iface->on_enter = uprobe_listener_on_enter;
// 	iface->on_leave = uprobe_listener_on_leave;
// }

// static void uprobe_listener_init(UprobeListener *self)
// {
// }

// static void frida_uprobe_listener_on_enter(_GumInvocationContext *ic,
// 					   void *data)
// {
// 	hook_entry *hook_entry = (struct hook_entry *)data;
// 	if (hook_entry->progs.size() == 0) {
// 		return;
// 	}
// 	GumInvocationContext *ctx;
// 	pt_regs regs;

// 	spdlog::trace("Handle uprobe at frida_uprobe_listener_on_enter");

// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	for (auto &prog : hook_entry->progs) {
// 		uint64_t ret_val;
// 		int res =
// 			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
// 		if (res < 0) {
// 			return;
// 		}
// 	}
// }

// static void frida_uprobe_listener_on_leave(_GumInvocationContext *ic,
// 					   void *data)
// {
// 	hook_entry *hook_entry = (struct hook_entry *)data;
// 	if (hook_entry->ret_progs.size() == 0) {
// 		return;
// 	}

// 	spdlog::trace("Handle uretprobe at frida_uprobe_listener_on_leave");
// 	pt_regs regs;
// 	GumInvocationContext *ctx;
// 	ctx = gum_interceptor_get_current_invocation();
// 	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
// 	for (auto &prog : hook_entry->ret_progs) {
// 		uint64_t ret_val;
// 		int res =
// 			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
// 		if (res < 0) {
// 			return;
// 		}
// 	}
// }

} // namespace bpftime

extern "C" {

// uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n, uint64_t *value)
// {
// 	GumInvocationContext *gum_ctx =
// 		gum_interceptor_get_current_invocation();
// 	if (gum_ctx == NULL) {
// 		return -EINVAL;
// 	}
// 	// ignore ctx;
// 	*value = (uint64_t)gum_cpu_context_get_nth_argument(
// 		gum_ctx->cpu_context, n);
// 	return 0;
// }

// uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value)
// {
// 	GumInvocationContext *gum_ctx =
// 		gum_interceptor_get_current_invocation();
// 	if (gum_ctx == NULL) {
// 		return -EOPNOTSUPP;
// 	}
// 	// ignore ctx;
// 	*value = (uint64_t)gum_invocation_context_get_return_value(gum_ctx);
// 	return 0;
// }

} // extern "C"
