#include "frida_internal_attach_entry.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "frida_attach_entry.hpp"
#include <dlfcn.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <frida_register_conversion.hpp>
using namespace bpftime::attach;

extern "C" uint64_t __bpftime_frida_attach_manager__replace_handler();
extern "C" void *__bpftime_frida_attach_manager__override_handler();

static void uprobe_listener_on_enter(GumInvocationContext *context,
				     gpointer user_data);
static void uprobe_listener_on_leave(GumInvocationContext *context,
				     gpointer user_data);

namespace
{

namespace fmt_lib = spdlog::fmt_lib;

const char *attach_type_to_string(int attach_type)
{
	switch (attach_type) {
	case ATTACH_UPROBE:
		return "uprobe";
	case ATTACH_URETPROBE:
		return "uretprobe";
	case ATTACH_UPROBE_OVERRIDE:
		return "uprobe_override";
	default:
		return "unknown";
	}
}

const char *gum_attach_return_to_string(int err)
{
	switch ((GumAttachReturn)err) {
	case GUM_ATTACH_OK:
		return "GUM_ATTACH_OK";
	case GUM_ATTACH_WRONG_SIGNATURE:
		return "GUM_ATTACH_WRONG_SIGNATURE";
	case GUM_ATTACH_ALREADY_ATTACHED:
		return "GUM_ATTACH_ALREADY_ATTACHED";
	case GUM_ATTACH_POLICY_VIOLATION:
		return "GUM_ATTACH_POLICY_VIOLATION";
	case GUM_ATTACH_WRONG_TYPE:
		return "GUM_ATTACH_WRONG_TYPE";
	default:
		return "GUM_ATTACH_UNKNOWN";
	}
}

std::string describe_attach_target(void *function)
{
	Dl_info info {};
	if (dladdr(function, &info) == 0) {
		return "symbol=<unresolved>";
	}

	std::string rendered;
	if (info.dli_fname != nullptr) {
		rendered = fmt_lib::format("module={}", info.dli_fname);
	}
	if (info.dli_sname != nullptr) {
		auto symbol = fmt_lib::format("symbol={}", info.dli_sname);
		if (info.dli_saddr != nullptr) {
			auto offset = (uintptr_t)function - (uintptr_t)info.dli_saddr;
			symbol = fmt_lib::format("{}+0x{:x}", symbol, offset);
		}
		if (!rendered.empty()) {
			rendered = fmt_lib::format("{}, {}", rendered, symbol);
		} else {
			rendered = symbol;
		}
	}
	if (rendered.empty()) {
		return "symbol=<unresolved>";
	}
	return rendered;
}

std::string format_target_bytes(void *function, size_t byte_count = 8)
{
	auto *bytes = reinterpret_cast<const uint8_t *>(function);
	std::string rendered;
	rendered.reserve(byte_count * 3);
	for (size_t i = 0; i < byte_count; i++) {
		if (i != 0) {
			rendered += ' ';
		}
		rendered += fmt_lib::format(
			"{:02x}", static_cast<unsigned int>(bytes[i]));
	}
	return rendered;
}

std::string build_frida_attach_failure_message(const char *operation,
					       void *function, int attach_type,
					       int err)
{
	return fmt_lib::format(
		"{} failed for attach_type={} at function 0x{:x} (err={}, {}, {}, first_bytes={}). "
		"Frida may reject very short functions or unsupported signatures; "
		"try compiling the target with -O0, adding __attribute__((noinline)), "
		"or attaching to a larger wrapper.",
		operation, attach_type_to_string(attach_type),
		(uintptr_t)function, err, gum_attach_return_to_string(err),
		describe_attach_target(function), format_target_bytes(function));
}

} // namespace

void frida_internal_attach_entry::ensure_listener(int attach_type)
{
	if (attach_type != ATTACH_UPROBE && attach_type != ATTACH_URETPROBE)
		return;
	auto **slot = attach_type == ATTACH_UPROBE ? &uprobe_listener :
						     &uretprobe_listener;
	if (*slot != nullptr)
		return;
	*slot = attach_type == ATTACH_UPROBE ?
			gum_make_probe_listener(uprobe_listener_on_enter, this,
						nullptr) :
			gum_make_call_listener(nullptr, uprobe_listener_on_leave,
					       this, nullptr);
	if (*slot == nullptr)
		throw std::runtime_error("Unable to create Frida listener");
	gum_interceptor_begin_transaction(interceptor);
	if (int err = gum_interceptor_attach(interceptor, function, *slot,
					     nullptr);
	    err < 0) {
		auto message = build_frida_attach_failure_message(
			"gum_interceptor_attach", function, attach_type, err);
		g_object_unref(*slot);
		*slot = nullptr;
		gum_interceptor_end_transaction(interceptor);
		SPDLOG_ERROR("{}", message);
		throw std::runtime_error(message);
	}
	gum_interceptor_end_transaction(interceptor);
}

frida_internal_attach_entry::frida_internal_attach_entry(
	void *function, int basic_attach_type, GumInterceptor *interceptor)
	: function(function)
{
	this->interceptor = interceptor;
	override_return_callback = nullptr;
	if (basic_attach_type == ATTACH_UPROBE ||
	    basic_attach_type == ATTACH_URETPROBE) {
		ensure_listener(basic_attach_type);
	} else if (basic_attach_type == ATTACH_UPROBE_OVERRIDE) {
		gum_interceptor_begin_transaction(interceptor);
		if (int err = gum_interceptor_replace(
			    interceptor, function,
			    (void *)__bpftime_frida_attach_manager__override_handler,
			    this, nullptr);
		    err < 0) {
			auto message = build_frida_attach_failure_message(
				"gum_interceptor_replace", function,
				basic_attach_type, err);
			gum_interceptor_end_transaction(interceptor);
			SPDLOG_ERROR("{}", message);
			throw std::runtime_error(message);
		}
		gum_interceptor_end_transaction(interceptor);
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
	for (auto *listener : { uprobe_listener, uretprobe_listener }) {
		if (listener != nullptr) {
			gum_interceptor_detach(interceptor, listener);
			g_object_unref(listener);
		}
	}
	if (!uprobe_listener && !uretprobe_listener) {
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

	SPDLOG_DEBUG("Setting current thread gum cpu context");
	current_thread_gum_cpu_context = ctx->cpu_context;

	hook_entry->run_filter_callback(regs);
	SPDLOG_DEBUG("Resetting current thread gum cpu context");
	current_thread_gum_cpu_context.reset();
	if (hook_entry->is_overrided) {
		auto value = (uintptr_t)hook_entry->user_ret;
		SPDLOG_DEBUG("Using override return value: {}", value);
		return (void *)value;
	} else {
		return func((void *)arg0, (void *)arg1, (void *)arg2,
			    (void *)arg3, (void *)arg4);
	}
}

static void uprobe_listener_on_enter(GumInvocationContext *ctx,
				     gpointer user_data)
{
	auto *hook_entry = static_cast<frida_internal_attach_entry *>(user_data);
	SPDLOG_TRACE("Handle uprobe at uprobe_listener_on_enter");
	bpftime::pt_regs regs;
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);

	SPDLOG_DEBUG("Setting current thread gum cpu context");
	current_thread_gum_cpu_context = ctx->cpu_context;

	hook_entry->iterate_uprobe_callbacks(regs);

	SPDLOG_DEBUG("Resetting current thread gum cpu context");
	current_thread_gum_cpu_context.reset();
}

static void uprobe_listener_on_leave(GumInvocationContext *ctx,
				     gpointer user_data)
{
	auto *hook_entry = static_cast<frida_internal_attach_entry *>(user_data);
	SPDLOG_TRACE("Handle uretprobe at uprobe_listener_on_leave");
	bpftime::pt_regs regs;
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	hook_entry->iterate_uretprobe_callbacks(regs);
}
