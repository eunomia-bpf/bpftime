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
#include <common/bpftime_config.hpp>
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

static int load_prog_and_helpers(bpftime_prog *prog, agent_config &config)
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

int bpf_attach_ctx::init_attach_ctx_from_handlers(agent_config &config)
{
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
		return -1;
	}
	return init_attach_ctx_from_handlers(manager, config);
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
				    bpf_perf_event_handler::bpf_event_type::
					    PERF_TYPE_TRACEPOINT) {
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
	spdlog::debug("Syscall callback {} {} {} {} {} {} {}", sys_nr, arg1,
		      arg2, arg3, arg4, arg5, arg6);
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
		const auto exec = [&](const bpftime_prog *prog) {
			spdlog::debug("Call {}", prog->prog_name());
			auto lctx = ctx;
			// Avoid polluting other ebpf programs..
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			assert(err >= 0);
		};
		for (const auto &item : sys_enter_progs[sys_nr]) {
			exec(item);
		}
		for (auto item : global_sys_enter_progs) {
			exec(item);
		}
	}
	int64_t ret = orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	if (!sys_exit_progs[sys_nr].empty() || !global_sys_exit_progs.empty()) {
		trace_event_raw_sys_exit ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.ret = ret;
		const auto exec = [&](const bpftime_prog *prog) {
			// Avoid polluting other ebpf programs..
			auto lctx = ctx;
			uint64_t ret;
			int err = prog->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			assert(err >= 0);
		};
		for (const auto &item : sys_exit_progs[sys_nr])
			exec(item);
		for (auto item : global_sys_exit_progs)
			exec(item);
	}
	return ret;
}

// create a attach context and progs from handlers
int bpf_attach_ctx::init_attach_ctx_from_handlers(
	const handler_manager *manager, agent_config &config)
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
				handler_prog_fds[v].emplace_back(i, prog);
			}
			spdlog::debug("Load prog fd={} name={}", i,
				      prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			spdlog::debug("bpf_map_handler found at {}", i);
		} else if (std::holds_alternative<bpf_perf_event_handler>(
				   handler)) {
			spdlog::debug("Will handle bpf_perf_events later...");

		} else if (std::holds_alternative<epoll_handler>(handler)) {
			spdlog::debug(
				"No extra operations needed for epoll_handler..");
		} else {
			spdlog::error("Unsupported handler type {}",
				      handler.index());
			return -1;
		}
	}
	for (const auto &[k, v] : handler_prog_fds) {
		for (auto y : v) {
			spdlog::debug(
				"Program fd {} attached to perf event handler {}",
				y.first, k);
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
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_FILTER:
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_REPLACE:
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_UPROBE:
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_URETPROBE:
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
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_FILTER: {
				auto progs = handler_prog_fds[i];
				if (progs.size() > 1) {
					spdlog::error(
						"Expected that a certain function could only be attached one filter, at perf event {}",
						i);
					return -E2BIG;
				}
				err = attach_manager->attach_filter_at(
					func_addr,
					[=](const pt_regs &regs) -> bool {
						uint64_t ret;
						progs[0].second
							->bpftime_prog_exec(
								(void *)&regs,
								sizeof(regs),
								&ret);
						return !ret;
					});
				if (err < 0)
					spdlog::error(
						"Failed to create filter for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_REPLACE: {
				auto progs = handler_prog_fds[i];
				if (progs.size() > 1) {
					spdlog::error(
						"Expected that a certain function could only be attached one replace, at perf event {}",
						i);
					return -E2BIG;
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
					spdlog::error(
						"Failed to create replace for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_UPROBE: {
				spdlog::debug(
					"Creating uprobe for perf event fd {}",
					i);
				auto progs = handler_prog_fds[i];
				spdlog::info(
					"Attached {} uprobe programs to function {:x}",
					progs.size(), (uintptr_t)func_addr);
				err = attach_manager->attach_uprobe_at(
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
					spdlog::error(
						"Failed to create uprobe for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_URETPROBE: {
				spdlog::debug(
					"Creating uretprobe for perf event fd {}",
					i);
				auto progs = handler_prog_fds[i];
				spdlog::info(
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
					spdlog::error(
						"Failed to create uretprobe for perf fd {}, err={}",
						i, err);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				PERF_TYPE_TRACEPOINT: {
				err = create_tracepoint(
					event_handler.tracepoint_id, i,
					manager);

				if (err < 0)
					spdlog::error(
						"Failed to create tracepoint for perf fd {}, err={}",
						i, err);
				assert(err >= 0);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				PERF_TYPE_SOFTWARE: {
				spdlog::debug(
					"Attaching software perf event, nothing need to do");
				err = i;
			}
			default:
				break;
			}
			spdlog::debug("Create attach event {} {} {} for {}", i,
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
	spdlog::debug("Destructor: bpf_attach_ctx");
}

int bpf_attach_ctx::create_tracepoint(int tracepoint_id, int perf_fd,
				      const handler_manager *manager)
{
	// Look up the corresponding tracepoint name like
	const auto &tp_table = get_global_syscall_tracepoint_table();
	const auto &[id_table, _] = get_global_syscall_id_table();
	if (auto itr = tp_table.find(tracepoint_id); itr != tp_table.end()) {
		spdlog::info("Creating tracepoint for tp name {}", itr->second);
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
				sys_exit_progs[syscall_id->second].push_back(p);
			spdlog::info(
				"Registered syscall exit hook for {} with perf fd {}",
				syscall_name, perf_fd);
			return perf_fd;
		} else if (name == GLOBAL_SYS_ENTER_NAME) {
			for (auto p : progs)
				global_sys_enter_progs.push_back(p);
			spdlog::info(
				"Registered global sys enter hook with perf fd {}",
				perf_fd);
			return perf_fd;
		} else if (name == GLOBAL_SYS_EXIT_NAME) {
			for (auto p : progs)
				global_sys_exit_progs.push_back(p);
			spdlog::info(
				"Registered global sys exit hook with perf fd {}",
				perf_fd);
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

// create a probe context
bpf_attach_ctx::bpf_attach_ctx(void)
	: attach_manager(std::make_unique<frida_attach_manager>())
{
	spdlog::debug("Initialzing frida gum");
	current_id = CURRENT_ID_OFFSET;
}
} // namespace bpftime
