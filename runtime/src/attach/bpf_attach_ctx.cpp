/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach/attach_manager/frida_attach_manager.hpp"
#include "bpftime.hpp"
#include "handler/epoll_handler.hpp"
#include <asm/unistd_64.h>
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

static int load_prog_and_helpers(bpftime_prog *prog, const agent_config &config)
{
	if (config.enable_kernel_helper_group) {
		bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog);
	}
	if (config.enable_ffi_helper_group) {
		bpftime_helper_group::get_ffi_helper_group()
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
	// Maintain perf_event fd -> [(prog fd,bpftime_prog*)]
	std::map<int, std::vector<std::pair<int, bpftime_prog *> > >
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
					    manager->get_handler(v))) {
					const auto &perf_handler =
						std::get<bpf_perf_event_handler>(
							manager->get_handler(
								v));
					if (perf_handler.enabled) {
						handler_prog_fds[v].emplace_back(
							i, prog);
						SPDLOG_DEBUG(
							"Program fd {} attached to perf event handler {}",
							i, v);
					} else {
						SPDLOG_INFO(
							"Ignore perf {} attached by prog fd {}. It's not enabled",
							v, i);
					}

				} else {
					spdlog::warn(
						"Program fd {} attached to a non-perf event handler {}",
						i, v);
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
			int err = -1;

			auto &event_handler =
				std::get<bpf_perf_event_handler>(handler);
			void *func_addr = nullptr;
			switch (event_handler.type) {
			case bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE:
			case bpf_event_type::BPF_TYPE_UREPLACE:
			case bpf_event_type::BPF_TYPE_UPROBE:
			case bpf_event_type::BPF_TYPE_URETPROBE:
				func_addr =
					attach_manager
						->resolve_function_addr_by_module_offset(
							event_handler
								._module_name
								.c_str(),
							event_handler.offset);
				break;
			default:
				break;
			}
			// attach base on events
			switch (event_handler.type) {
			case bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE: {
				SPDLOG_DEBUG(
					"Creating filter for perf event fd {}",
					i);
				if (func_addr == nullptr) {
					return -ENOENT;
				}
				auto progs = handler_prog_fds[i];
				if (progs.size() > 1) {
					SPDLOG_ERROR(
						"Expected that a certain function could only be attached one filter, at perf event {}",
						i);
					return -E2BIG;
				} else if (progs.empty()) {
					SPDLOG_ERROR(
						"Perf event {} doesn't have any attached & enabled programs",
						i);
					return -ENOENT;
				}
				err = attach_manager->attach_uprobe_override_at(
					func_addr, [=](const pt_regs &regs) {
						uint64_t ret;
						progs[0].second
							->bpftime_prog_exec(
								(void *)&regs,
								sizeof(regs),
								&ret);
					});
				if (err < 0)
					SPDLOG_ERROR(
						"Failed to create filter for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_event_type::BPF_TYPE_UREPLACE: {
				SPDLOG_DEBUG(
					"Creating replace for perf event fd {}",
					i);
				if (func_addr == nullptr) {
					return -ENOENT;
				}
				auto progs = handler_prog_fds[i];
				if (progs.size() > 1) {
					SPDLOG_ERROR(
						"Expected that a certain function could only be attached one replace, at perf event {}",
						i);
					return -E2BIG;
				} else if (progs.empty()) {
					SPDLOG_ERROR(
						"Perf event {} doesn't have any attached & enabled programs",
						i);
					return -ENOENT;
				}
				err = attach_manager->attach_replace_at(
					func_addr,
					[=](const pt_regs &regs) -> uint64_t {
						uint64_t ret;
						progs[0].second
							->bpftime_prog_exec(
								(void *)&regs,
								sizeof(regs),
								&ret);
						return ret;
					});
				if (err < 0)
					SPDLOG_ERROR(
						"Failed to create replace for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_event_type::BPF_TYPE_UPROBE: {
				SPDLOG_DEBUG(
					"Creating uprobe for perf event fd {}",
					i);
				if (func_addr == nullptr) {
					return -ENOENT;
				}
				auto progs = handler_prog_fds[i];
				SPDLOG_INFO(
					"Attached {} uprobe programs to function {:x}",
					progs.size(), (uintptr_t)func_addr);
				err = attach_manager->attach_uprobe_at(
					func_addr, [=](const pt_regs &regs) {
						SPDLOG_TRACE(
							"Uprobe triggered");
						uint64_t ret;
						for (auto &[k, prog] : progs) {
							SPDLOG_TRACE(
								"Calling ebpf programs in uprobe callback");
							prog->bpftime_prog_exec(
								(void *)&regs,
								sizeof(regs),
								&ret);
						}
					});
				if (err < 0)
					SPDLOG_ERROR(
						"Failed to create uprobe for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_event_type::BPF_TYPE_URETPROBE: {
				SPDLOG_DEBUG(
					"Creating uretprobe for perf event fd {}",
					i);
				if (func_addr == nullptr) {
					return -ENOENT;
				}
				auto progs = handler_prog_fds[i];
				SPDLOG_INFO(
					"Attached {} uretprobe programs to function {:x}",
					progs.size(), (uintptr_t)func_addr);
				err = attach_manager->attach_uretprobe_at(
					func_addr, [=](const pt_regs &regs) {
						uint64_t ret;
						for (auto &[k, prog] : progs) {
							prog->bpftime_prog_exec(
								(void *)&regs,
								sizeof(regs),
								&ret);
						}
					});
				if (err < 0)
					SPDLOG_ERROR(
						"Failed to create uretprobe for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_event_type::PERF_TYPE_TRACEPOINT: {
				SPDLOG_DEBUG(
					"Creating tracepoint for perf event fd {}",
					i);
				err = create_tracepoint(
					event_handler.tracepoint_id, i,
					manager);

				if (err < 0)
					SPDLOG_ERROR(
						"Failed to create tracepoint for perf fd {}, err={}",
						i, err);
				assert(err >= 0);
				break;
			}
			case bpf_event_type::PERF_TYPE_SOFTWARE: {
				SPDLOG_DEBUG(
					"Attaching software perf event, nothing need to do");
				err = i;
				break;
			}
			default:
				spdlog::warn("Unexpected bpf_event_type: {}",
					     (int)event_handler.type);
				break;
			}
			SPDLOG_DEBUG("Create attach event {} {} {} for {}", i,
				     event_handler._module_name,
				     event_handler.offset, err);
			if (err < 0) {
				return err;
			}
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
	: attach_manager(std::make_unique<frida_attach_manager>())
{
	SPDLOG_DEBUG("Initialzing frida gum");
	current_id = CURRENT_ID_OFFSET;
}

} // namespace bpftime
