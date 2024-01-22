/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach/attach_manager/frida_attach_manager.hpp"
#include "bpftime.hpp"
#include "handler/epoll_handler.hpp"
#include <unistd.h>
#include <cerrno>
#include <map>
#include <memory>
#include <syscall_table.hpp>
#include <bpf_attach_ctx.hpp>
#include <bpftime_shm_internal.hpp>
#include <bpftime_prog.hpp>
#include "bpftime_config.hpp"
#include <spdlog/spdlog.h>
#include <handler/perf_event_handler.hpp>
#include <bpftime_helper_group.hpp>
#include <handler/handler_manager.hpp>
#include <attach/attach_internal.hpp>
#include <utility>
#include <variant>
#include <sys/resource.h>

namespace bpftime
{

int bpf_attach_ctx::create_tracepoint(int tracepoint_id, int perf_fd,
				      const handler_manager *manager)
{
	// Look up the corresponding tracepoint name like
	const auto &tp_table = get_global_syscall_tracepoint_table();
	const auto &[id_table, _] = get_global_syscall_id_table();
	if (auto itr = tp_table.find(tracepoint_id); itr != tp_table.end()) {
		SPDLOG_INFO("Creating tracepoint for tp name {}", itr->second);
		// Lookup the corresponding bpf progs by
		// brute force
		std::vector<const bpftime_prog *> progs;

		for (std::size_t i = 0; i < manager->size(); i++) {
			if (manager->is_allocated(i) &&
			    std::holds_alternative<bpf_prog_handler>(
				    manager->get_handler(i))) {
				auto &prog = std::get<bpf_prog_handler>(
					manager->get_handler(i));
				for (auto v : prog.attach_fds) {
					if (v.first == perf_fd) {
						progs.push_back(
							this->progs[i].get());
						assert(progs.back());
					}
				}
			}
		}
		if (progs.empty()) {
			SPDLOG_ERROR("bpf_link for perf event {} not found",
				     perf_fd);
			return perf_fd;
		}
		const auto &name = itr->second;
		if (name.starts_with("sys_enter_")) {
			auto syscall_name = name.substr(10);
			auto syscall_id = id_table.find(syscall_name);
			if (syscall_id == id_table.end()) {
				SPDLOG_ERROR("Syscall id not found for name {}",
					     syscall_name);
				return -1;
			}
			for (auto p : progs)
				sys_enter_progs[syscall_id->second].push_back(
					p);
			SPDLOG_INFO(
				"Registered syscall enter hook for {} with perf fd {}",
				syscall_name, perf_fd);
			return perf_fd;
		} else if (name.starts_with("sys_exit_")) {
			auto syscall_name = name.substr(9);
			auto syscall_id = id_table.find(syscall_name);
			if (syscall_id == id_table.end()) {
				SPDLOG_ERROR("Syscall id not found for name {}",
					     syscall_name);
				return -1;
			}
			for (auto p : progs)
				sys_exit_progs[syscall_id->second].push_back(p);
			SPDLOG_INFO(
				"Registered syscall exit hook for {} with perf fd {}",
				syscall_name, perf_fd);
			return perf_fd;
		} else if (name == GLOBAL_SYS_ENTER_NAME) {
			for (auto p : progs)
				global_sys_enter_progs.push_back(p);
			SPDLOG_INFO(
				"Registered global sys enter hook with perf fd {}",
				perf_fd);
			return perf_fd;
		} else if (name == GLOBAL_SYS_EXIT_NAME) {
			for (auto p : progs)
				global_sys_exit_progs.push_back(p);
			SPDLOG_INFO(
				"Registered global sys exit hook with perf fd {}",
				perf_fd);
			return perf_fd;
		} else {
			SPDLOG_ERROR("Unexpected syscall tracepoint name {}",
				     name);
			return -1;
		}
	} else {
		SPDLOG_ERROR("Unsupported tracepoint id: {}", tracepoint_id);
		return -1;
	}
}

// Check whether there is a syscall trace program. Use the global
// handler manager
bool bpf_attach_ctx::check_exist_syscall_trace_program()
{
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
		return false;
	}
	return this->check_exist_syscall_trace_program(manager);
}

// Check whether there is a syscall trace program
bool bpf_attach_ctx::check_exist_syscall_trace_program(
	const handler_manager *manager)
{
	for (size_t i = 0; i < manager->size(); i++) {
		if (manager->is_allocated(i)) {
			auto &handler = manager->get_handler(i);
			if (std::holds_alternative<bpf_perf_event_handler>(
				    handler)) {
				auto &perf_event_handler =
					std::get<bpf_perf_event_handler>(
						handler);
				if (perf_event_handler.type ==
				    bpf_event_type::PERF_TYPE_TRACEPOINT) {
					const auto &tp_table =
						get_global_syscall_tracepoint_table();
					if (tp_table.find(
						    perf_event_handler
							    .tracepoint_id) !=
					    tp_table.end()) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

// Check whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
bool bpf_attach_ctx::check_syscall_trace_setup(int pid)
{
	return shm_holder.global_shared_memory.check_syscall_trace_setup(pid);
}

// Set whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
void bpf_attach_ctx::set_syscall_trace_setup(int pid, bool whether)
{
	shm_holder.global_shared_memory.set_syscall_trace_setup(pid, whether);
}

int64_t bpf_attach_ctx::run_syscall_hooker(int64_t sys_nr, int64_t arg1,
					   int64_t arg2, int64_t arg3,
					   int64_t arg4, int64_t arg5,
					   int64_t arg6)
{
	if (sys_nr == __NR_exit_group || sys_nr == __NR_exit)
		return orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	SPDLOG_DEBUG("Syscall callback {} {} {} {} {} {} {}", sys_nr, arg1,
		     arg2, arg3, arg4, arg5, arg6);
	bool is_overrided = false;
	uint64_t user_ret = 0;
	uint64_t user_ret_ctx = 0;
	curr_thread_override_return_callback =
		override_return_set_callback([&](uint64_t ctx, uint64_t v) {
			is_overrided = true;
			user_ret = v;
			user_ret_ctx = ctx;
		});
	if (!sys_enter_progs[sys_nr].empty() ||
	    !global_sys_enter_progs.empty()) {
		trace_event_raw_sys_enter ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.args[0] = arg1;
		ctx.args[1] = arg2;
		ctx.args[2] = arg3;
		ctx.args[3] = arg4;
		ctx.args[4] = arg5;
		ctx.args[5] = arg6;
		for (const auto prog : sys_enter_progs[sys_nr]) {
			SPDLOG_DEBUG("Call {}", prog->prog_name());
			auto lctx = ctx;
			// Avoid polluting other ebpf programs..
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			SPDLOG_DEBUG("ret {}", ret);
		}
		for (const auto prog : global_sys_enter_progs) {
			SPDLOG_DEBUG("Call {}", prog->prog_name());
			auto lctx = ctx;
			// Avoid polluting other ebpf programs..
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			SPDLOG_DEBUG("ret {}", ret);
		}
	}
	if (is_overrided) {
		curr_thread_override_return_callback.reset();
		return user_ret;
	}
	SPDLOG_DEBUG("exec original syscall");
	int64_t ret = orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	if (!sys_exit_progs[sys_nr].empty() || !global_sys_exit_progs.empty()) {
		trace_event_raw_sys_exit ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.ret = ret;
		for (const auto prog : sys_exit_progs[sys_nr]) {
			SPDLOG_DEBUG("Call {}", prog->prog_name());
			auto lctx = ctx;
			// Avoid polluting other ebpf programs..
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			SPDLOG_DEBUG("ret {}", ret);
		}
		for (const auto prog : global_sys_exit_progs) {
			SPDLOG_DEBUG("Call {}", prog->prog_name());
			auto lctx = ctx;
			// Avoid polluting other ebpf programs..
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			SPDLOG_DEBUG("ret {}", ret);
		}
	}
	if (is_overrided) {
		curr_thread_override_return_callback.reset();
		return user_ret;
	}
	return ret;
}

} // namespace bpftime
