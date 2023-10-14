#include "bpftime.hpp"
#include "frida-gum.h"
#include "handler/epoll_handler.hpp"
#include <asm/unistd_64.h>
#include <filesystem>
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
#include <variant>
static std::string get_executable_path()
{
	char exec_path[PATH_MAX] = { 0 };
	ssize_t len =
		readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
	if (len != -1) {
		exec_path[len] = '\0'; // Null-terminate the string
		spdlog::info("Executable path: {}", exec_path);
	} else {
		spdlog::error("Error retrieving executable path: {}", errno);
	}
	return exec_path;
}

namespace bpftime
{
static void *
resolve_function_addr(bpf_attach_ctx &ctx,
		      const bpftime::bpf_perf_event_handler &event_handler)
{
	void *function = nullptr;
	const char *module_name = event_handler._module_name.c_str();
	std::string exec_path = get_executable_path();
	void *module_base_addr;

	if (std::filesystem::equivalent(exec_path, module_name)) {
		// the current process
		module_base_addr = ctx.module_get_base_addr("");
	} else {
		module_base_addr = ctx.module_get_base_addr(
			event_handler._module_name.c_str());
	}
	// find function
	if (!module_base_addr) {
		spdlog::error("module {} not found",
			      event_handler._module_name.c_str());
		return nullptr;
	}
	function = (void *)((char *)module_base_addr + event_handler.offset);
	return function;
}

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
	spdlog::debug("Syscall callback");
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
	// First, we create programs
	for (std::size_t i = 0; i < manager->size(); i++) {
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
			spdlog::debug("Load prog {} {}", i, prog_handler.name);
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
	// Second, we create bpf perf event handlers
	for (std::size_t i = 0; i < manager->size(); i++) {
		if (!manager->is_allocated(i)) {
			continue;
		}
		auto &handler = manager->get_handler(i);

		if (std::holds_alternative<bpf_perf_event_handler>(handler)) {
			int fd = -1;
			auto &event_handler =
				std::get<bpf_perf_event_handler>(handler);
			void *function = nullptr;
			if (event_handler.type !=
				    bpf_perf_event_handler::bpf_event_type::
					    PERF_TYPE_TRACEPOINT &&
			    event_handler.type !=
				    bpf_perf_event_handler::bpf_event_type::
					    PERF_TYPE_SOFTWARE) {
				function = resolve_function_addr(*this,
								 event_handler);
				if (!function) {
					spdlog::error(
						"Function not found {} {}",
						event_handler._module_name,
						event_handler.offset);
					errno = ENOENT;
					return -1;
				}
			}
			// attach base on events
			switch (event_handler.type) {
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_FILTER: {
				fd = create_filter(function, i);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_REPLACE: {
				fd = create_replace(function, i);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_UPROBE: {
				fd = create_uprobe(function, i, false);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				BPF_TYPE_URETPROBE: {
				spdlog::debug(
					"Creating uretprobe for perf event fd {}",
					i);
				fd = create_uprobe(function, i, true);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				PERF_TYPE_TRACEPOINT: {
				fd = create_tracepoint(
					event_handler.tracepoint_id, i,
					manager);
				assert(fd >= 0);
				break;
			}
			case bpf_perf_event_handler::bpf_event_type::
				PERF_TYPE_SOFTWARE: {
				spdlog::debug(
					"Attaching software perf event, nothing need to do");
				fd = i;
			}
			default:
				break;
			}
			spdlog::debug("Create attach event {} {} {} for {}", i,
				     event_handler._module_name,
				     event_handler.offset, fd);
			if (fd < 0) {
				return fd;
			}
		}
	}
	// attach the progs to the fds
	return attach_progs_in_manager(manager);
}

int bpf_attach_ctx::attach_progs_in_manager(const handler_manager *manager)
{
	// attach the progs to the fds
	for (auto &prog : progs) {
		int id = prog.first;
		// get the handler and find the attach information
		auto &prog_handler =
			std::get<bpf_prog_handler>(manager->get_handler(id));
		for (auto fd : prog_handler.attach_fds) {
			spdlog::debug("Attaching prog {} to fd {}", id, fd);
			attach_prog(prog.second.get(), fd);
		}
	}
	return 0;
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
		spdlog::error("Unable to find function {} {}", name,
			      __FUNCTION__);
	}
	return NULL;
}

void *bpf_attach_ctx::module_find_export_by_name(const char *module_name,
						 const char *symbol_name)
{
	return (void *)(uintptr_t)gum_module_find_export_by_name(module_name,
								 symbol_name);
}

void *bpf_attach_ctx::module_get_base_addr(const char *module_name)
{
	gum_module_load(module_name, nullptr);
	return (void *)gum_module_find_base_address(module_name);
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

int bpf_attach_ctx::create_tracepoint(int tracepoint_id, int perf_fd,
				      const handler_manager *manager)
{
	// Look up the corresponding tracepoint name like
	const auto &tp_table = get_global_syscall_tracepoint_table();
	const auto &[id_table, _] = get_global_syscall_id_table();
	if (auto itr = tp_table.find(tracepoint_id); itr != tp_table.end()) {
		spdlog::info("Creating tracepoint for tp name {}", itr->second);
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
// create a probe context
bpf_attach_ctx::bpf_attach_ctx(void)
{
	spdlog::debug("Initialzing frida gum");
	gum_init_embedded();

	interceptor = gum_interceptor_obtain();

	current_id = CURRENT_ID_OFFSET;
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
		uprobe_listener_get_type(), NULL);

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
} // namespace bpftime
