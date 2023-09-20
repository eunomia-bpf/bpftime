#include "bpftime_handler.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <cstring>
#include <optional>
#include <ostream>
#include <unistd.h>
#include "bpftime.hpp"
#include "bpftime_internal.h"
#include <frida-gum.h>
#include <syscall_table.hpp>
#include <variant>
#include <vector>
#include <spdlog/spdlog.h>

using namespace std;
using namespace bpftime;

namespace bpftime
{

#ifndef __u64
#define __u64 uint64_t
#endif

#if defined(__x86_64__) || defined(_M_X64)

struct pt_regs {
	uint64_t r15;
	uint64_t r14;
	uint64_t r13;
	uint64_t r12;
	uint64_t bp;
	uint64_t bx;
	uint64_t r11;
	uint64_t r10;
	uint64_t r9;
	uint64_t r8;
	uint64_t ax;
	uint64_t cx;
	uint64_t dx;
	uint64_t si;
	uint64_t di;
	uint64_t orig_ax;
	uint64_t ip;
	uint64_t cs;
	uint64_t flags;
	uint64_t sp;
	uint64_t ss;
};

#elif defined(__aarch64__) || defined(_M_ARM64)
struct pt_regs {
	__u64 regs[31];
	__u64 sp;
	__u64 pc;
	__u64 pstate;
};
#elif defined(__arm__) || defined(_M_ARM)
struct pt_regs {
	uint32_t uregs[18];
};
#else
#error "Unsupported architecture"
#endif

uint64_t __bpftime_trace_handler(struct pt_regs *regs);

// the bpf program will be called instead of the function execution.
// for attach replace
uint64_t __frida_bpftime_replace_handler(void);
void *__frida_bpftime_filter_handler();

// create a probe context
bpf_attach_ctx::bpf_attach_ctx(void)
{
	gum_init_embedded();

	interceptor = gum_interceptor_obtain();

	current_id = CURRENT_ID_OFFSET;
}

// replace the function for the old program
int bpf_attach_ctx::replace_func(void *new_function, void *target_function,
				 void *data)
{
	/* Transactions are optional but improve performance with multiple
	 * hooks. */
	gum_interceptor_begin_transaction(interceptor);

	auto res = gum_interceptor_replace(interceptor, target_function,
					   new_function, data, NULL);
	if (res < 0) {
		return res;
	}
	/*
	 * ^
	 * |
	 * This is using replace(), but there's also attach() which can be used
	 * to hook functions without any knowledge of argument types, calling
	 * convention, etc. It can even be used to put a probe in the middle of
	 * a function.
	 */
	gum_interceptor_end_transaction(interceptor);
	return 0;
}

int bpf_attach_ctx::create_tracepoint(int tracepoint_id, int perf_fd,
				      const handler_manager *manager)
{
	// Look up the corresponding tracepoint name like
	const auto &tp_table = get_global_syscall_tracepoint_table();
	const auto &[id_table, _] = get_global_syscall_id_table();
	if (auto itr = tp_table.find(tracepoint_id); itr != tp_table.end()) {
		spdlog::info("Creating tracepoint for tp name {}",
			      itr->second);
		// I'm lazy. So I just lookup the corresponding bpf progs by
		// brute force

#warning Inefficient algorithm here. Remeber to rewrite it in the future
		std::vector<const bpftime_prog *> progs;

		for (std::size_t i = 0; i < manager->size(); i++) {
			if (manager->is_allocated(i) &&
			    std::holds_alternative<bpf_prog_handler>(
				    manager->get_handler(i))) {
				auto &prog = std::get<bpf_prog_handler>(
					manager->get_handler(i));
				for (auto v : prog.attach_fds) {
					if (v == perf_fd) {
						progs.push_back(
							this->progs[i].get());
						assert(progs.back());
					}
				}
			}
		}
		if (progs.empty()) {
			spdlog::error("bpf_link for perf event {} not found",
				      perf_fd);
			return perf_fd;
		}
		const auto &name = itr->second;
		if (name.starts_with("sys_enter_")) {
			auto syscall_name = name.substr(10);
			auto syscall_id = id_table.find(syscall_name);
			if (syscall_id == id_table.end()) {
				spdlog::error(
					"Syscall id not found for name {}",
					syscall_name);
				return -1;
			}
			for (auto p : progs)
				sys_enter_progs[syscall_id->second].push_back(
					p);
			spdlog::info(
				"Registered syscall enter hook for {} with perf fd {}",
				syscall_name, perf_fd);
			return perf_fd;
		} else if (name.starts_with("sys_exit_")) {
			auto syscall_name = name.substr(9);
			auto syscall_id = id_table.find(syscall_name);
			if (syscall_id == id_table.end()) {
				spdlog::error(
					"Syscall id not found for name {}",
					syscall_name);
				return -1;
			}
			for (auto p : progs)
				sys_enter_progs[syscall_id->second].push_back(
					p);
			spdlog::info(
				"Registered syscall exit hook for {} with perf fd {}",
				syscall_name, perf_fd);
			return perf_fd;
		} else {
			spdlog::error("Unexpected syscall tracepoint name {}",
				      name);
			return -1;
		}
	} else {
		spdlog::error("Unsupported tracepoint id: {}", tracepoint_id);
		return -1;
	}
}

int bpf_attach_ctx::create_replace_with_handler(int id,
						bpftime_hook_entry_type type,
						void *function,
						void *handler_func)
{
	spdlog::info("create_replace_with_handler {:x}", (uintptr_t)function);
	if (handler_func == NULL) {
		handler_func = (void *)__frida_bpftime_replace_handler;
	}
	if (hook_entry_index.find(id) != hook_entry_index.end()) {
		// already has a id
		return -1;
	}
	if (hook_entry_table.find(function) != hook_entry_table.end()) {
		// already has a hook
		return -1;
	}
	auto iter = hook_entry_table.emplace(function, hook_entry{});
	if (!iter.second) {
		return -1;
	}
	auto entry = iter.first;
	entry->second.id = id;
	entry->second.type = type;
	entry->second.hook_func = function;
	entry->second.handler_function = handler_func;
	hook_entry_index[id] = function;
	auto res = replace_func(handler_func, function, &entry->second);
	if (res < 0) {
		spdlog::error("replace_func failed");
		hook_entry_table.erase(function);
		hook_entry_index.erase(id);
		return res;
	}
	return id;
}

// the bpf program will be called instead of the function execution.
int bpf_attach_ctx::create_replace(void *function)
{
	// Split the increment of volatile current_id to make clang happy
	int current_id = this->current_id;
	this->current_id = this->current_id + 1;
	return create_replace(function, current_id);
}

int bpf_attach_ctx::create_replace(void *function, int id)
{
	return create_replace_with_handler(
		id, BPFTIME_REPLACE, function,
		(void *)__frida_bpftime_replace_handler);
}

int bpf_attach_ctx::create_filter(void *function)
{
	// Split the increment of volatile current_id to make clang happy
	int current_id = this->current_id;
	this->current_id = this->current_id + 1;
	return create_filter(function, current_id);
}

int bpf_attach_ctx::create_filter(void *function, int id)
{
	return create_replace_with_handler(
		id, BPFTIME_REPLACE, function,
		(void *)__frida_bpftime_filter_handler);
}

int bpf_attach_ctx::revert_func(void *target_function)
{
	gum_interceptor_revert(interceptor, target_function);
	return 0;
}

int bpf_attach_ctx::destory_attach(int id)
{
	auto function = hook_entry_index.find(id);
	if (function == hook_entry_index.end()) {
		return -1;
	}
	auto entry = hook_entry_table.find(function->second);
	if (entry == hook_entry_table.end()) {
		return -1;
	}
	switch (entry->second.type) {
	case BPFTIME_REPLACE: {
		if (entry->second.hook_func != nullptr) {
			revert_func(entry->second.hook_func);
		}
		entry->second.progs.clear();
		hook_entry_index.erase(id);
		hook_entry_table.erase(function->second);
		return 0;
	}
	case BPFTIME_UPROBE: {
		if (entry->second.uretprobe_id == id) {
			hook_entry_index.erase(entry->second.uretprobe_id);
			entry->second.uretprobe_id = -1;
			entry->second.ret_progs.clear();
		}
		if (entry->second.id == id) {
			hook_entry_index.erase(entry->second.id);
			entry->second.id = -1;
			entry->second.progs.clear();
		}
		if (entry->second.listener != nullptr && entry->second.id < 0 &&
		    entry->second.uretprobe_id < 0) {
			// detach the listener when no one is using it
			gum_interceptor_detach(interceptor,
					       entry->second.listener);
			gum_free(entry->second.listener);
			hook_entry_table.erase(function->second);
		}
		return 0;
	}
	default:
		break;
	}
	return 0;
}

int bpf_attach_ctx::attach_prog(const bpftime_prog *prog, int id)
{
	// cannot find the attach target
	if (hook_entry_index.find(id) == hook_entry_index.end()) {
		return -1;
	}
	auto function = hook_entry_index[id];
	if (hook_entry_table.find(function) == hook_entry_table.end()) {
		return -1;
	}
	auto &entry = hook_entry_table[function];
	switch (entry.type) {
	case BPFTIME_REPLACE: {
		// replace handler can only have one prog
		if (entry.progs.size() > 0) {
			return -1;
		}
		entry.progs.insert(prog);
		break;
	}
	case BPFTIME_UPROBE: {
		if (entry.uretprobe_id == id) {
			entry.ret_progs.insert(prog);
		} else {
			entry.progs.insert(prog);
		}
		break;
	}
	default:
		return -1;
	}
	return 0;
}

int bpf_attach_ctx::detach(const bpftime_prog *prog)
{
	for (auto m : hook_entry_table) {
		if (m.second.progs.find(prog) != m.second.progs.end()) {
			m.second.progs.erase(prog);
		}
		if (m.second.ret_progs.find(prog) != m.second.ret_progs.end()) {
			m.second.ret_progs.erase(prog);
		}
		break;
	}
	return 0;
}

bpf_attach_ctx::~bpf_attach_ctx()
{
	std::vector<int> id_to_remove = {};
	for (auto &m : hook_entry_table) {
		id_to_remove.push_back(m.second.id);
	}
	for (auto i : id_to_remove) {
		destory_attach(i);
	}
}

void *bpf_attach_ctx::module_find_export_by_name(const char *module_name,
						 const char *symbol_name)
{
	return (void *)(uintptr_t)gum_module_find_export_by_name(module_name,
								 symbol_name);
}

void *bpf_attach_ctx::module_get_base_addr(const char *module_name)
{
	return (void *)gum_module_find_base_address(module_name);
}

// find the function by name in current process
void *bpf_attach_ctx::find_function_by_name(const char *name)
{
	char *error_msg;
	void *addr;
	// try to find addr from frida
	addr = (void *)gum_find_function(name);
	if (addr) {
		return addr;
	}
	// try to find addr from module
	addr = module_find_export_by_name(NULL, name);
	if (addr) {
		return addr;
	}
	if (addr == NULL) {
		spdlog::error("Unable to finc function {} {}", name,
			      __FUNCTION__);
	}
	return NULL;
}

#if defined(__x86_64__) || defined(_M_X64)

static inline void
convert_gum_cpu_context_to_pt_regs(const _GumX64CpuContext &context,
				   pt_regs &regs)
{
	regs.ip = context.rip;
	regs.r15 = context.r15;
	regs.r14 = context.r14;
	regs.r13 = context.r13;
	regs.r12 = context.r12;
	regs.r11 = context.r11;
	regs.r10 = context.r10;
	regs.r9 = context.r9;
	regs.r8 = context.r8;
	regs.di = context.rdi;
	regs.si = context.rsi;
	regs.bp = context.rbp;
	regs.sp = context.rsp;
	regs.bx = context.rbx;
	regs.dx = context.rdx;
	regs.cx = context.rcx;
	regs.ax = context.rax;
}

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumX64CpuContext &context)
{
	context.rip = regs.ip;
	context.r15 = regs.r15;
	context.r14 = regs.r14;
	context.r13 = regs.r13;
	context.r12 = regs.r12;
	context.r11 = regs.r11;
	context.r10 = regs.r10;
	context.r9 = regs.r9;
	context.r8 = regs.r8;
	context.rdi = regs.di;
	context.rsi = regs.si;
	context.rbp = regs.bp;
	context.rsp = regs.sp;
	context.rbx = regs.bx;
	context.rdx = regs.dx;
	context.rcx = regs.cx;
	context.rax = regs.ax;
}

#elif defined(__aarch64__) || defined(_M_ARM64)
static inline void
convert_gum_cpu_context_to_pt_regs(const _GumArm64CpuContext &context,
				   pt_regs &regs)
{
	memcpy(&regs.regs, &context.x, sizeof(context.x));
	regs.regs[29] = context.fp;
	regs.regs[30] = context.lr;
	regs.sp = context.sp;
	regs.pc = context.pc;
	regs.pstate = context.nzcv;
}

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumArm64CpuContext &context)
{
	memcpy(&context.x, &regs.regs, sizeof(context.x));
	context.fp = regs.regs[29];
	context.lr = regs.regs[30];
	context.sp = regs.sp;
	context.pc = regs.pc;
	context.nzcv = regs.pstate;
}
#elif defined(__arm__) || defined(_M_ARM)
static inline void
convert_gum_cpu_context_to_pt_regs(const _GumArmCpuContext &context,
				   pt_regs &regs)
{
	for (size_t i = 0; i < std::size(context.r); i++) {
		regs.uregs[i] = context.r[i];
	}
	regs.uregs[8] = context.r8;
	regs.uregs[9] = context.r9;
	regs.uregs[10] = context.r10;
	regs.uregs[11] = context.r11;
	regs.uregs[12] = context.r12;
	regs.uregs[13] = context.sp;
	regs.uregs[14] = context.lr;
	regs.uregs[15] = context.pc;
	regs.uregs[16] = context.cpsr;
	regs.uregs[17] = 0;
}

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumArmCpuContext &context)
{
	for (size_t i = 0; i < std::size(context.r); i++) {
		context.r[i] = regs.uregs[i];
	}
	context.r8 = regs.uregs[8];
	context.r9 = regs.uregs[9];
	context.r10 = regs.uregs[10];
	context.r11 = regs.uregs[11];
	context.r12 = regs.uregs[12];
	context.sp = regs.uregs[13];
	context.lr = regs.uregs[14];
	context.pc = regs.uregs[15];
	context.cpsr = regs.uregs[16];
}
#else
#error "Unsupported architecture"
#endif

uint64_t __frida_bpftime_replace_handler()
{
	GumInvocationContext *ctx;
	struct hook_entry *hook_entry;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	hook_entry = (struct hook_entry *)
		gum_invocation_context_get_replacement_data(ctx);
	auto ptr = hook_entry->progs.begin();
	if (ptr == hook_entry->progs.end()) {
		return 0;
	}
	const bpftime_prog *prog = *ptr;
	if (!prog) {
		return 0;
	}
	uint64_t ret_val;
	int res = prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
	if (res < 0) {
		return 0;
	}
	gum_invocation_context_replace_return_value(ctx, (gpointer)ret_val);
	return ret_val;
}

void *__frida_bpftime_filter_handler()
{
	GumInvocationContext *ctx;
	struct hook_entry *hook_entry;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	hook_entry = (struct hook_entry *)
		gum_invocation_context_get_replacement_data(ctx);
	auto ptr = hook_entry->progs.begin();
	if (ptr == hook_entry->progs.end()) {
		return 0;
	}
	const bpftime_prog *prog = *ptr;
	if (!prog) {
		return 0;
	}
	// naive implementation
	// TODO: is there any better ways?
	auto arg0 = gum_invocation_context_get_nth_argument(ctx, 0);
	auto arg1 = gum_invocation_context_get_nth_argument(ctx, 1);
	auto arg2 = gum_invocation_context_get_nth_argument(ctx, 2);
	auto arg3 = gum_invocation_context_get_nth_argument(ctx, 3);
	auto arg4 = gum_invocation_context_get_nth_argument(ctx, 4);
	ffi_func func = (ffi_func)ctx->function;
	uint64_t op_code;
	int res = prog->bpftime_prog_exec(&regs, sizeof(regs), &op_code);
	if (res < 0) {
		return 0;
	}
	// printf("filter op code: %" PRId64 " ret: %" PRIuPTR "\n",
	// op_code,
	//        (uintptr_t)hook_entry->ret_val);
	if (op_code == OP_SKIP) {
		// drop the function call and not proceed
		return hook_entry->ret_val;
	} else if (op_code != OP_RESUME) {
		g_printerr("Invalid op code %" PRId64 "\n", op_code);
		return (void *)(uintptr_t)op_code;
	}

	// recover the origin function
	return func((void *)arg0, (void *)arg1, (void *)arg2, (void *)arg3,
		    (void *)arg4);
}

// replace the function for the old program
int bpf_attach_ctx::add_listener(GumInvocationListener *listener,
				 void *target_function, void *data)
{
	/* Transactions are optional but improve performance with multiple
	 * hooks. */
	gum_interceptor_begin_transaction(interceptor);
	auto res = gum_interceptor_attach(interceptor, target_function,
					  listener, data);
	if (res < 0) {
		return res;
	}
	/*
	 * ^
	 * |
	 * This is using replace(), but there's also attach() which can be used
	 * to hook functions without any knowledge of argument types, calling
	 * convention, etc. It can even be used to put a probe in the middle of
	 * a function.
	 */
	gum_interceptor_end_transaction(interceptor);
	return 0;
}

typedef struct _UprobeListener UprobeListener;

struct _UprobeListener {
	GObject parent;
};

static void uprobe_listener_iface_init(gpointer g_iface, gpointer iface_data);

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
	hook_entry *hook_entry = (struct hook_entry *)
		gum_invocation_context_get_listener_function_data(ic);
	if (hook_entry->progs.size() == 0) {
		return;
	}
	GumInvocationContext *ctx;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	for (auto &prog : hook_entry->progs) {
		uint64_t ret_val;
		int res =
			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
		if (res < 0) {
			return;
		}
	}
}

static void uprobe_listener_on_leave(GumInvocationListener *listener,
				     GumInvocationContext *ic)
{
	hook_entry *hook_entry = (struct hook_entry *)
		gum_invocation_context_get_listener_function_data(ic);
	if (hook_entry->ret_progs.size() == 0) {
		return;
	}

	pt_regs regs;
	GumInvocationContext *ctx;
	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	for (auto &prog : hook_entry->ret_progs) {
		uint64_t ret_val;
		int res =
			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
		if (res < 0) {
			return;
		}
	}
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

static void frida_uprobe_listener_on_enter(_GumInvocationContext *ic,
					   void *data)
{
	hook_entry *hook_entry = (struct hook_entry *)data;
	if (hook_entry->progs.size() == 0) {
		return;
	}
	GumInvocationContext *ctx;
	pt_regs regs;

	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	for (auto &prog : hook_entry->progs) {
		uint64_t ret_val;
		int res =
			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
		if (res < 0) {
			return;
		}
	}
}

static void frida_uprobe_listener_on_leave(_GumInvocationContext *ic,
					   void *data)
{
	hook_entry *hook_entry = (struct hook_entry *)data;
	if (hook_entry->ret_progs.size() == 0) {
		return;
	}

	pt_regs regs;
	GumInvocationContext *ctx;
	ctx = gum_interceptor_get_current_invocation();
	convert_gum_cpu_context_to_pt_regs(*ctx->cpu_context, regs);
	for (auto &prog : hook_entry->ret_progs) {
		uint64_t ret_val;
		int res =
			prog->bpftime_prog_exec(&regs, sizeof(regs), &ret_val);
		if (res < 0) {
			return;
		}
	}
}

// attach to a function in the object.
int bpf_attach_ctx::create_uprobe(void *function, int id, bool retprobe)
{
	if (hook_entry_index.find(id) != hook_entry_index.end()) {
		// already has a id
		return -1;
	}
	if (hook_entry_table.find(function) != hook_entry_table.end()) {
		// already has a hook
		auto &entry = hook_entry_table[function];
		if (entry.type != BPFTIME_UPROBE) {
			// other type of hook
			return -1;
		}
		if (retprobe) {
			if (entry.uretprobe_id > 0) {
				// already has a uretprobe
				return -1;
			}
			entry.uretprobe_id = id;
		} else {
			if (entry.id < 0) {
				// already has a uprobe
				return -1;
			}
			entry.id = id;
		}
		hook_entry_index[id] = function;
		return id;
	}
	auto iter = hook_entry_table.emplace(function, hook_entry{});
	if (!iter.second) {
		return -1;
	}
	auto entry = iter.first;
	if (retprobe) {
		entry->second.uretprobe_id = id;
	} else {
		entry->second.id = id;
	}
	entry->second.type = BPFTIME_UPROBE;
	entry->second.hook_func = function;
	auto entry_ptr = &entry->second;
	entry->second.listener = (GumInvocationListener *)g_object_new(
		EXAMPLE_TYPE_LISTENER, NULL);

	// gum_make_call_listener(frida_uprobe_listener_on_enter,
	// 		       frida_uprobe_listener_on_leave,
	// 		       entry_ptr, NULL);
	int res = add_listener(entry->second.listener, function, entry_ptr);
	if (res < 0) {
		hook_entry_table.erase(function);
		return res;
	}
	hook_entry_index[id] = function;
	return id;
}

} // namespace bpftime

extern "C" {

uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n, uint64_t *value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EINVAL;
	}
	// ignore ctx;
	*value = (uint64_t)gum_cpu_context_get_nth_argument(
		gum_ctx->cpu_context, n);
	return 0;
}

uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	// ignore ctx;
	*value = (uint64_t)gum_invocation_context_get_return_value(gum_ctx);
	return 0;
}

uint64_t bpftime_get_retval(void)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	return (uintptr_t)gum_invocation_context_get_return_value(gum_ctx);
}

uint64_t bpftime_set_retval(uint64_t value)
{
	GumInvocationContext *gum_ctx =
		gum_interceptor_get_current_invocation();
	if (gum_ctx == NULL) {
		return -EOPNOTSUPP;
	}
	struct hook_entry *entry = (struct hook_entry *)
		gum_invocation_context_get_replacement_data(gum_ctx);
	entry->ret_val = (void *)(uintptr_t)value;
	gum_invocation_context_replace_return_value(gum_ctx,
						    (gpointer)((size_t)value));
	return 0;
}

} // extern "C"
