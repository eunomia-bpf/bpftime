#include "spdlog/spdlog.h"
#include "syscall_trace_attach_private_data.hpp"
#include <cerrno>
#include <iterator>
#include <optional>
#include <syscall_trace_attach_impl.hpp>

#ifdef __linux__
#include <asm/unistd.h>  // For architecture-specific syscall numbers
#endif

namespace bpftime
{
namespace attach
{
std::optional<syscall_trace_attach_impl *> global_syscall_trace_attach_impl;

#if defined(__x86_64__)
struct syscall_pt_regs {
	uint64_t r15, r14, r13, r12, bp, bx, r11, r10, r9, r8, ax, cx, dx,
		si, di, orig_ax, ip, cs, flags, sp, ss;
};
#elif defined(__aarch64__)
struct syscall_pt_regs {
	uint64_t regs[31], sp, pc, pstate;
};
#endif

int64_t syscall_trace_attach_impl::dispatch_syscall(int64_t sys_nr,
						    int64_t arg1, int64_t arg2,
						    int64_t arg3, int64_t arg4,
						    int64_t arg5, int64_t arg6,
						    int64_t user_ip,
						    int64_t user_sp,
						    int64_t user_bp)
{
	if (sys_nr < 0 ||
	    sys_nr >= static_cast<int64_t>(std::size(sys_enter_callbacks))) {
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				    user_ip, user_sp, user_bp);
	}
	const bool run_enter = !sys_enter_callbacks[sys_nr].empty() ||
			       !global_enter_callbacks.empty();
	const bool run_exit = !sys_exit_callbacks[sys_nr].empty() ||
			      !global_exit_callbacks.empty();
	if (!run_enter && !run_exit) {
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				    user_ip, user_sp, user_bp);
	}
	attach_callback_scope callback_scope;
	if (!callback_scope.entered())
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				    user_ip, user_sp, user_bp);
// Exit syscall may cause bugs since it's not return to userspace
#ifdef __linux__
	if (sys_nr == __NR_exit_group || sys_nr == __NR_exit)
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				    user_ip, user_sp, user_bp);
#endif
	SPDLOG_DEBUG("Syscall callback {} {} {} {} {} {} {}", sys_nr, arg1,
		     arg2, arg3, arg4, arg5, arg6);
	// Keep the std::function target within its small-object buffer. A syscall
	// can be issued while libc holds its malloc lock, so allocating here would
	// recursively enter malloc and deadlock the target process.
	struct override_state {
		bool is_overrided = false;
		uint64_t user_ret = 0;
		uint64_t user_ret_ctx = 0;
	} override_state;
	auto run_callback = [](syscall_trace_attach_entry *prog,
			       const auto &ctx) noexcept {
		try {
			auto ctx_copy = ctx;
			uint64_t callback_ret = 0;
			int err = prog->cb(&ctx_copy, sizeof(ctx_copy),
					   &callback_ret);
			SPDLOG_DEBUG("ret {}, err {}", callback_ret, err);
		} catch (...) {
		}
	};
	auto run_enter_callbacks = [&](const auto &callbacks) noexcept {
		for (auto prog : callbacks) {
			if (prog->attach_type == ATTACH_SYSCALL_TRACE) {
				trace_event_raw_sys_enter ctx = {};
				ctx.id = sys_nr;
				ctx.args[0] = arg1;
				ctx.args[1] = arg2;
				ctx.args[2] = arg3;
				ctx.args[3] = arg4;
				ctx.args[4] = arg5;
				ctx.args[5] = arg6;
				run_callback(prog, ctx);
			} else if (prog->kprobe_abi !=
				   syscall_kprobe_abi::syscall_wrapper) {
#if defined(__x86_64__)
				syscall_pt_regs ctx = {};
				ctx.si = arg1;
				ctx.dx = arg2;
				ctx.cx = arg3;
				ctx.r8 = arg4;
				ctx.r9 = arg5;
				ctx.ip = user_ip;
				ctx.sp = user_sp;
				ctx.bp = user_bp;
				ctx.cs = 0x33;
				ctx.ss = 0x2b;
#elif defined(__aarch64__)
				syscall_pt_regs ctx = {};
				ctx.regs[1] = arg1;
				ctx.regs[2] = arg2;
				ctx.regs[3] = arg3;
				ctx.regs[4] = arg4;
				ctx.regs[5] = arg5;
				ctx.regs[29] = user_bp;
				ctx.sp = user_sp;
				ctx.pc = user_ip;
#endif
				run_callback(prog, ctx);
			} else {
#if defined(__x86_64__)
				syscall_pt_regs inner = {};
				inner.di = arg1;
				inner.si = arg2;
				inner.dx = arg3;
				inner.r10 = arg4;
				inner.r8 = arg5;
				inner.r9 = arg6;
				syscall_pt_regs outer = {};
				outer.di = reinterpret_cast<uintptr_t>(&inner);
				outer.orig_ax = sys_nr;
				outer.ax = sys_nr;
				outer.ip = user_ip;
				outer.sp = user_sp;
				outer.bp = user_bp;
				outer.cs = 0x33;
				outer.ss = 0x2b;
#elif defined(__aarch64__)
				syscall_pt_regs inner = {};
				inner.regs[0] = arg1;
				inner.regs[1] = arg2;
				inner.regs[2] = arg3;
				inner.regs[3] = arg4;
				inner.regs[4] = arg5;
				inner.regs[5] = arg6;
				syscall_pt_regs outer = {};
				outer.regs[0] = reinterpret_cast<uintptr_t>(&inner);
				outer.regs[29] = user_bp;
				outer.sp = user_sp;
				outer.pc = user_ip;
#endif
				run_callback(prog, outer);
			}
		}
	};
	auto run_exit_callbacks = [&](const auto &callbacks,
				      int64_t ret) noexcept {
		for (auto prog : callbacks) {
			if (prog->attach_type == ATTACH_SYSCALL_TRACE) {
				trace_event_raw_sys_exit ctx = {};
				ctx.id = sys_nr;
				ctx.ret = ret;
				run_callback(prog, ctx);
			} else {
				syscall_pt_regs ctx = {};
#if defined(__x86_64__)
				ctx.ax = ret;
				ctx.orig_ax = sys_nr;
				ctx.ip = user_ip;
				ctx.sp = user_sp;
				ctx.bp = user_bp;
				ctx.cs = 0x33;
				ctx.ss = 0x2b;
#elif defined(__aarch64__)
				ctx.regs[0] = ret;
				ctx.regs[29] = user_bp;
				ctx.sp = user_sp;
				ctx.pc = user_ip;
#endif
				run_callback(prog, ctx);
			}
		}
	};

	if (run_enter) {
		curr_thread_override_return_callback =
			override_return_set_callback([state = &override_state](
						     uint64_t ctx, uint64_t v) {
				state->is_overrided = true;
				state->user_ret = v;
				state->user_ret_ctx = ctx;
			});
		run_enter_callbacks(sys_enter_callbacks[sys_nr]);
		run_enter_callbacks(global_enter_callbacks);
		curr_thread_override_return_callback.reset();
		if (override_state.is_overrided) {
			return override_state.user_ret;
		}
	}
	SPDLOG_DEBUG("executing original syscall");
	int64_t ret = orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				   user_ip, user_sp, user_bp);
	if (run_exit) {
		curr_thread_override_return_callback =
			override_return_set_callback([state = &override_state](
						     uint64_t ctx, uint64_t v) {
				state->is_overrided = true;
				state->user_ret = v;
				state->user_ret_ctx = ctx;
			});
		run_exit_callbacks(sys_exit_callbacks[sys_nr], ret);
		run_exit_callbacks(global_exit_callbacks, ret);
		curr_thread_override_return_callback.reset();
		if (override_state.is_overrided) {
			return override_state.user_ret;
		}
	}
	return ret;
}

int syscall_trace_attach_impl::detach_by_id(int id)
{
	SPDLOG_DEBUG("Detaching syscall trace attach entry {}", id);
	if (auto itr = attach_entries.find(id); itr != attach_entries.end()) {
		const auto &ent = itr->second;
		if (ent->is_enter && ent->sys_nr == -1) {
			global_enter_callbacks.erase(ent.get());
		} else if (!ent->is_enter && ent->sys_nr == -1) {
			global_exit_callbacks.erase(ent.get());
		} else if (ent->is_enter) {
			sys_enter_callbacks[ent->sys_nr].erase(ent.get());
		} else if (!ent->is_enter) {
			sys_exit_callbacks[ent->sys_nr].erase(ent.get());
		} else {
			SPDLOG_ERROR("Unreachable branch reached!");
			return -EINVAL;
		}
		attach_entries.erase(itr);
		return 0;
	} else {
		SPDLOG_ERROR("Invalid attach id {}", id);
		return -ENOENT;
	}
}
int syscall_trace_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	if (attach_type != ATTACH_SYSCALL_TRACE &&
	    attach_type != ATTACH_SYSCALL_KPROBE &&
	    attach_type != ATTACH_SYSCALL_KRETPROBE) {
		SPDLOG_ERROR(
			"Unsupported attach type {} by syscall trace attach impl",
			attach_type);
		return -ENOTSUP;
	}
	try {
		auto &priv_data =
			dynamic_cast<const syscall_trace_attach_private_data &>(
				private_data);
		if (priv_data.sys_nr >= (int)std::size(sys_enter_callbacks) ||
		    priv_data.sys_nr < -1) {
			SPDLOG_ERROR("Invalid sys nr {}", priv_data.sys_nr);
			return -EINVAL;
		}
		bool is_enter = attach_type == ATTACH_SYSCALL_KPROBE ||
				(attach_type == ATTACH_SYSCALL_TRACE &&
				 priv_data.is_enter);
		auto ent_ptr = std::make_unique<syscall_trace_attach_entry>(
				syscall_trace_attach_entry{
					.cb = cb,
					.sys_nr = priv_data.sys_nr,
					.is_enter = is_enter,
					.attach_type = attach_type,
					.kprobe_abi = priv_data.kprobe_abi });
		auto raw_ptr = ent_ptr.get();
		int id = allocate_id();
		attach_entries[id] = std::move(ent_ptr);
		if (is_enter) {
			if (priv_data.sys_nr == -1)
				global_enter_callbacks.insert(raw_ptr);
			else
				sys_enter_callbacks[priv_data.sys_nr].insert(
					raw_ptr);
		} else {
			if (priv_data.sys_nr == -1)
				global_exit_callbacks.insert(raw_ptr);
			else
				sys_exit_callbacks[priv_data.sys_nr].insert(
					raw_ptr);
		}
		return id;
	} catch (const std::bad_cast &ex) {
		SPDLOG_ERROR(
			"Syscall trace attach manager expected a private data of type syscall_trace_attach_private_data: {}",
			ex.what());
		return -EINVAL;
	}
}

extern "C" int64_t _bpftime__syscall_dispatcher(int64_t sys_nr, int64_t arg1,
						int64_t arg2, int64_t arg3,
						int64_t arg4, int64_t arg5,
						int64_t arg6, int64_t user_ip,
						int64_t user_sp,
						int64_t user_bp)
{
	SPDLOG_DEBUG("Call syscall dispatcher: {} {}, {}, {}, {}, {}, {}",
		     sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	return global_syscall_trace_attach_impl.value()->dispatch_syscall(
		sys_nr, arg1, arg2, arg3, arg4, arg5, arg6, user_ip, user_sp,
		user_bp);
}

extern "C" void
_bpftime__setup_syscall_hooker_callback(syscall_hooker_func_t *hooker)
{
	assert(global_syscall_trace_attach_impl.has_value());
	auto impl = global_syscall_trace_attach_impl.value();
	impl->set_original_syscall_function(*hooker);
	SPDLOG_DEBUG(
		"Saved original syscall hooker (original syscall function): {:x}",
		(uintptr_t)*hooker);
	*hooker = _bpftime__syscall_dispatcher;
	SPDLOG_DEBUG("Set syscall hooker to {:x}", (uintptr_t)*hooker);
}

} // namespace attach
} // namespace bpftime
