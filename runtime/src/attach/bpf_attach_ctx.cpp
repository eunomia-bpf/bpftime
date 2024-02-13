/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include "frida_attach_utils.hpp"
#include "handler/epoll_handler.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include <unistd.h>
#include <cerrno>
#include <cstdint>
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
#include <tuple>
#include <utility>
#include <variant>
#include <sys/resource.h>
#include <frida_uprobe_attach_internal.hpp>
#include <frida_uprobe_attach_impl.hpp>
#include <frida_attach_private_data.hpp>
extern "C" uint64_t bpftime_set_retval(uint64_t value);
namespace bpftime
{

static int load_prog_and_helpers(bpftime_prog *prog, const agent_config &config)
{
	if (config.enable_kernel_helper_group) {
		bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog);
	}
	if (config.enable_ufunc_helper_group) {
		bpftime_helper_group::get_ufunc_helper_group()
			.add_helper_group_to_prog(prog);
	}
	if (config.enable_shm_maps_helper_group) {
		bpftime_helper_group::get_shm_maps_helper_group()
			.add_helper_group_to_prog(prog);
	}
	return prog->bpftime_prog_load(config.jit_enabled);
}

int bpf_attach_ctx::init_attach_ctx_from_handlers(const agent_config &config)
{
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
		return -1;
	}
	return init_attach_ctx_from_handlers(manager, config);
}

// create a attach context and progs from handlers
int bpf_attach_ctx::init_attach_ctx_from_handlers(
	const handler_manager *manager, const agent_config &config)
{
	// Maintain perf_event fd -> [(prog fd, bpftime_prog*, attach cookie)]
	std::map<int, std::vector<std::tuple<int, bpftime_prog *,
					     std::optional<uint64_t> > > >
		handler_prog_fds;
	// First, we create programs
	for (std::size_t i = 0; i < manager->size(); i++) {
		// skip uninitialized handlers
		if (!manager->is_allocated(i)) {
			continue;
		}
		auto &handler = manager->get_handler(i);
		// load the bpf prog
		if (std::holds_alternative<bpf_prog_handler>(handler)) {
			auto &prog_handler =
				std::get<bpf_prog_handler>(handler);
			const ebpf_inst *insns = prog_handler.insns.data();
			size_t cnt = prog_handler.insns.size();
			const char *name = prog_handler.name.c_str();
			progs[i] = std::make_unique<bpftime_prog>(insns, cnt,
								  name);
			bpftime_prog *prog = progs[i].get();
			int res = load_prog_and_helpers(prog, config);
			if (res < 0) {
				return res;
			}
			for (auto v : prog_handler.attach_fds) {
				if (std::holds_alternative<
					    bpf_perf_event_handler>(
					    manager->get_handler(v.first))) {
					const auto &perf_handler =
						std::get<bpf_perf_event_handler>(
							manager->get_handler(
								v.first));
					if (perf_handler.enabled) {
						handler_prog_fds[v.first]
							.emplace_back(i, prog,
								      v.second);
						SPDLOG_DEBUG(
							"Program fd {} attached to perf event handler {}, enable cookie = {}, cookie value = {}",
							i, v.first,
							v.second.has_value(),
							v.second.value_or(0));
					} else {
						SPDLOG_INFO(
							"Ignore perf {} attached by prog fd {}. It's not enabled",
							v.first, i);
					}

				} else {
					spdlog::warn(
						"Program fd {} attached to a non-perf event handler {}",
						i, v.first);
				}
			}
			SPDLOG_DEBUG("Load prog fd={} name={}", i,
				     prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			SPDLOG_DEBUG("bpf_map_handler found at {}", i);
		} else if (std::holds_alternative<bpf_perf_event_handler>(
				   handler)) {
			SPDLOG_DEBUG("Will handle bpf_perf_events later...");

		} else if (std::holds_alternative<epoll_handler>(handler) ||
			   std::holds_alternative<bpf_link_handler>(handler)) {
			SPDLOG_DEBUG(
				"No extra operations needed for epoll_handler/bpf link/btf..");
		} else {
			SPDLOG_ERROR("Unsupported handler type for handler {}",
				     handler.index());
			return -1;
		}
	}

	// Second, we create bpf perf event handlers
	for (std::size_t i = 0; i < manager->size(); i++) {
		if (!manager->is_allocated(i)) {
			continue;
		}
		auto &handler = manager->get_handler(i);

		if (std::holds_alternative<bpf_perf_event_handler>(handler)) {
			auto &event_handler =
				std::get<bpf_perf_event_handler>(handler);
			std::unique_ptr<attach::attach_private_data> priv_data;
			attach::base_attach_impl *attach_impl;
			if (event_handler.type ==
				    bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE ||
			    event_handler.type ==
				    bpf_event_type::BPF_TYPE_UPROBE ||
			    event_handler.type ==
				    bpf_event_type::BPF_TYPE_URETPROBE ||
			    event_handler.type ==
				    bpf_event_type::BPF_TYPE_UREPLACE) {
				SPDLOG_DEBUG("Attaching uprobe series type {}",
					     event_handler.type);
				void *func_addr = attach::
					resolve_function_addr_by_module_offset(
						event_handler._module_name
							.c_str(),
						event_handler.offset);
				priv_data = std::make_unique<
					attach::frida_attach_private_data>();
				if (int err = priv_data->initialize_from_string(
					    std::to_string(
						    (uintptr_t)func_addr));
				    err < 0) {
					SPDLOG_ERROR(
						"Unable to initialize private data: {}",
						err);
					return err;
				}
				attach_impl = &get_uprobe_attach_impl();
			} else if (event_handler.type ==
				   bpf_event_type::PERF_TYPE_TRACEPOINT) {
				priv_data = std::make_unique<
					attach::syscall_trace_attach_private_data>();
				if (int err = priv_data->initialize_from_string(
					    std::to_string(
						    event_handler
							    .tracepoint_id));
				    err < 0) {
					SPDLOG_ERROR(
						"Unable to initialize private data: {}",
						err);
					return err;
				}
				attach_impl = &get_syscall_attach_impl();
			} else if (event_handler.type ==
				   bpf_event_type::PERF_TYPE_SOFTWARE) {
				SPDLOG_DEBUG(
					"Attaching software perf event, nothing need to do");
			} else {
				spdlog::warn("Unexpected bpf_event_type: {}",
					     (int)event_handler.type);
			}
			bool is_ureplace = event_handler.type ==
					   bpf_event_type::BPF_TYPE_UREPLACE;
			auto progs = handler_prog_fds[i];
			for (auto tup : progs) {
				auto prog = std::get<1>(tup);
				auto cookie = std::get<2>(tup);
				int id = attach_impl->create_attach_with_ebpf_callback(
					[=](void *mem, size_t mem_size,
					    uint64_t *ret) -> int {
						current_thread_bpf_cookie =
							cookie;
						int err =
							prog->bpftime_prog_exec(
								(void *)mem,
								mem_size, ret);
						if (is_ureplace && err >= 0)
							bpftime_set_retval(
								*ret);
						return err;
					},
					*priv_data,
					is_ureplace ?
						(int)bpf_event_type::
							BPF_TYPE_UPROBE_OVERRIDE :
						(int)event_handler.type);
				if (id < 0) {
					SPDLOG_ERROR(
						"Unable to attach type {}: {}",
						(int)event_handler.type, id);
					return id;
				}
			}
			SPDLOG_DEBUG("Create attach event {} {} {} for {}", i,
				     event_handler._module_name,
				     event_handler.offset, err);
		}
	}
	return 0;
}

bpf_attach_ctx::~bpf_attach_ctx()
{
	SPDLOG_DEBUG("Destructor: bpf_attach_ctx");
}

// create a probe context
bpf_attach_ctx::bpf_attach_ctx(void)
	: frida_uprobe_attach_impl(
		  std::make_unique<attach::frida_attach_impl>()),
	  syscall_trace_attach_impl(
		  std::make_unique<attach::syscall_trace_attach_impl>())
{
	current_id = CURRENT_ID_OFFSET;
}

} // namespace bpftime
