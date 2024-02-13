/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
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
#include <utility>
#include <variant>
#include <sys/resource.h>

namespace bpftime
{

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

} // namespace bpftime
