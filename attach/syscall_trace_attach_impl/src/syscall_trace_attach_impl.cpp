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

int64_t syscall_trace_attach_impl::dispatch_syscall(int64_t sys_nr,
						    int64_t arg1, int64_t arg2,
						    int64_t arg3, int64_t arg4,
						    int64_t arg5, int64_t arg6)
{
// Exit syscall may cause bugs since it's not return to userspace
#ifdef __linux__
	if (sys_nr == __NR_exit_group || sys_nr == __NR_exit)
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
#endif
	SPDLOG_DEBUG("Syscall callback {} {} {} {} {} {} {}", sys_nr, arg1,
		     arg2, arg3, arg4, arg5, arg6);
	// Indicate whether the return value is overridden
	bool is_overrided = false;
	uint64_t user_ret = 0;
	uint64_t user_ret_ctx = 0;
	curr_thread_override_return_callback =
		override_return_set_callback([&](uint64_t ctx, uint64_t v) {
			is_overrided = true;
			user_ret = v;
			user_ret_ctx = ctx;
		});

	if (!sys_enter_callbacks[sys_nr].empty() ||
	    !global_enter_callbacks.empty()) {
		trace_event_raw_sys_enter ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.args[0] = arg1;
		ctx.args[1] = arg2;
		ctx.args[2] = arg3;
		ctx.args[3] = arg4;
		ctx.args[4] = arg5;
		ctx.args[5] = arg6;
		for (auto prog : sys_enter_callbacks[sys_nr]) {
			auto ctx_copy = ctx;
			uint64_t ret;
			int err = prog->cb(&ctx_copy, sizeof(ctx_copy), &ret);
			SPDLOG_DEBUG("ret {}, err {}", ret, err);
		}
		for (auto prog : global_enter_callbacks) {
			auto ctx_copy = ctx;
			uint64_t ret;
			int err = prog->cb(&ctx_copy, sizeof(ctx_copy), &ret);
			SPDLOG_DEBUG("ret {}, err {}", ret, err);
		}
	}
	curr_thread_override_return_callback.reset();
	if (is_overrided) {
		return user_ret;
	}
	curr_thread_override_return_callback =
		override_return_set_callback([&](uint64_t ctx, uint64_t v) {
			is_overrided = true;
			user_ret = v;
			user_ret_ctx = ctx;
		});
	SPDLOG_DEBUG("executing original syscall");
	int64_t ret = orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	if (!sys_exit_callbacks[sys_nr].empty() ||
	    !global_exit_callbacks.empty()) {
		trace_event_raw_sys_exit ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.ret = ret;
		for (auto prog : sys_exit_callbacks[sys_nr]) {
			auto ctx_copy = ctx;
			uint64_t ret;
			int err = prog->cb(&ctx_copy, sizeof(ctx_copy), &ret);
			SPDLOG_DEBUG("ret {}, err {}", ret, err);
		}
		for (const auto prog : global_exit_callbacks) {
			auto ctx_copy = ctx;
			uint64_t ret;
			int err = prog->cb(&ctx_copy, sizeof(ctx_copy), &ret);
			SPDLOG_DEBUG("ret {}, err {}", ret, err);
		}
	}
	curr_thread_override_return_callback.reset();
	if (is_overrided) {
		return user_ret;
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
	if (attach_type != ATTACH_SYSCALL_TRACE) {
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
		auto ent_ptr = std::make_unique<syscall_trace_attach_entry>(
			syscall_trace_attach_entry{
				.cb = cb,
				.sys_nr = priv_data.sys_nr,
				.is_enter = priv_data.is_enter });
		auto raw_ptr = ent_ptr.get();
		int id = allocate_id();
		attach_entries[id] = std::move(ent_ptr);
		if (priv_data.is_enter) {
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
						int64_t arg6)
{
	SPDLOG_DEBUG("Call syscall dispatcher: {} {}, {}, {}, {}, {}, {}",
		     sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	return global_syscall_trace_attach_impl.value()->dispatch_syscall(
		sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
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
