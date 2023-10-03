#include "bpftime.hpp"
#include "bpftime_handler.hpp"
#include "syscall_table.hpp"
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <filesystem>
#include <ostream>
#include <sched.h>
#include <variant>
#include <spdlog/spdlog.h>

using namespace bpftime;

// memory region for maps and prog info
// Use union so that the destructor of bpftime_shm won't be called automatically
union bpftime_shm_holder {
	bpftime_shm global_shared_memory;
	bpftime_shm_holder()
	{
		// Use placement new, which will not allocate memory, but just
		// call the constructor
		new (&global_shared_memory) bpftime_shm;
	}
	~bpftime_shm_holder()
	{
	}
};
static bpftime_shm_holder shm_holder;

static __attribute__((destructor(1))) void __destroy_bpftime_shm_holder()
{
	shm_holder.global_shared_memory.~bpftime_shm();
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
	if (!sys_enter_progs[sys_nr].empty()) {
		trace_event_raw_sys_enter ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.args[0] = arg1;
		ctx.args[1] = arg2;
		ctx.args[2] = arg3;
		ctx.args[3] = arg4;
		ctx.args[4] = arg5;
		ctx.args[5] = arg6;
		for (const auto &item : sys_enter_progs[sys_nr]) {
			// Avoid polluting other ebpf programs..
			auto lctx = ctx;
			uint64_t ret;
			int err = item->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			assert(err >= 0);
		}
	}
	int64_t ret = orig_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	if (!sys_exit_progs[sys_nr].empty()) {
		trace_event_raw_sys_exit ctx;
		memset(&ctx, 0, sizeof(ctx));
		ctx.id = sys_nr;
		ctx.ret = ret;
		for (const auto &item : sys_exit_progs[sys_nr]) {
			// Avoid polluting other ebpf programs..
			auto lctx = ctx;
			uint64_t ret;
			int err = item->bpftime_prog_exec(&lctx, sizeof(lctx),
							  &ret);
			assert(err >= 0);
		}
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
			spdlog::info("Load prog {} {}", i, prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			spdlog::info("bpf_map_handler found at {}", i);
		} else if (std::holds_alternative<bpf_perf_event_handler>(
				   handler)) {
			spdlog::info("Will handle bpf_perf_events later...");

		} else {
			spdlog::error("Unsupported handler type");
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
				    PERF_TYPE_TRACEPOINT) {
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
			default:
				break;
			}
			spdlog::info("Create attach event {} {} {} for {}", i,
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
			spdlog::info("Attaching prog {} to fd {}", id, fd);
			attach_prog(prog.second.get(), fd);
		}
	}
	return 0;
}
uint32_t bpftime_shm::bpf_map_value_size(int fd) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.get_value_size();
}
const void *bpftime_shm::bpf_map_lookup_elem(int fd, const void *key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return nullptr;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_lookup_elem(key);
}

long bpftime_shm::bpf_update_elem(int fd, const void *key, const void *value,
				  uint64_t flags) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_update_elem(key, value, flags);
}

long bpftime_shm::bpf_delete_elem(int fd, const void *key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_delete_elem(key);
}

int bpftime_shm::bpf_map_get_next_key(int fd, const void *key,
				      void *next_key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.bpf_map_get_next_key(key, next_key);
}

int bpftime_shm::add_uprobe(int pid, const char *name, uint64_t offset,
			    bool retprobe, size_t ref_ctr_off)
{
	int fd = open_fake_fd();
	manager->set_handler(
		fd,
		bpftime::bpf_perf_event_handler{ false, offset, pid, name,
						 ref_ctr_off, segment },
		segment);
	return fd;
}
int bpftime_shm::add_tracepoint(int pid, int32_t tracepoint_id)
{
	int fd = open_fake_fd();
	manager->set_handler(fd,
			     bpftime::bpf_perf_event_handler(pid, tracepoint_id,
							     segment),
			     segment);
	return fd;
}
int bpftime_shm::attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	if (!is_perf_fd(perf_fd) || !is_prog_fd(bpf_fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_prog_handler>(
		manager->get_handler(bpf_fd));
	handler.add_attach_fd(perf_fd);
	return 0;
}

int bpftime_shm::attach_enable(int fd) const
{
	if (!is_perf_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_perf_event_handler>(
		manager->get_handler(fd));
	handler.enable();
	return 0;
}

namespace bpftime
{
bpftime::agent_config &bpftime_get_agent_config()
{
	return shm_holder.global_shared_memory.get_agent_config();
}
} // namespace bpftime

// Check whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
bool bpftime_shm::check_syscall_trace_setup(int pid)
{
	return syscall_installed_pids->contains(pid);
}
// Set whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
void bpftime_shm::set_syscall_trace_setup(int pid, bool whether)
{
	if (whether) {
		syscall_installed_pids->insert(pid);
	} else {
		syscall_installed_pids->erase(pid);
	}
}

int bpftime_link_create(int prog_fd, int target_fd)
{
	return shm_holder.global_shared_memory.add_bpf_link(prog_fd, target_fd);
}

int bpftime_progs_create(const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
{
	return shm_holder.global_shared_memory.add_bpf_prog(
		insn, insn_cnt, prog_name, prog_type);
}

int bpftime_maps_create(const char *name, bpftime::bpf_map_attr attr)
{
	return shm_holder.global_shared_memory.add_bpf_map(name, attr);
}
uint32_t bpftime_map_value_size(int fd)
{
	return shm_holder.global_shared_memory.bpf_map_value_size(fd);
}

const void *bpftime_map_lookup_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_map_lookup_elem(fd, key);
}

long bpftime_map_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags)
{
	return shm_holder.global_shared_memory.bpf_update_elem(fd, key, value,
							       flags);
}

long bpftime_map_delete_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_delete_elem(fd, key);
}
int bpftime_map_get_next_key(int fd, const void *key, void *next_key)
{
	return shm_holder.global_shared_memory.bpf_map_get_next_key(fd, key,
								    next_key);
}

int bpftime_uprobe_create(int pid, const char *name, uint64_t offset,
			  bool retprobe, size_t ref_ctr_off)
{
	return shm_holder.global_shared_memory.add_uprobe(
		pid, name, offset, retprobe, ref_ctr_off);
}

int bpftime_tracepoint_create(int pid, int32_t tp_id)
{
	return shm_holder.global_shared_memory.add_tracepoint(pid, tp_id);
}

int bpftime_attach_enable(int fd)
{
	return shm_holder.global_shared_memory.attach_enable(fd);
}

int bpftime_attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	return shm_holder.global_shared_memory.attach_perf_to_bpf(perf_fd,
								  bpf_fd);
}

void bpftime_close(int fd)
{
	shm_holder.global_shared_memory.close_fd(fd);
}
int bpftime_map_get_info(int fd, bpftime::bpf_map_attr *out_attr,
			 const char **out_name, int *type)
{
	if (!shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_map_handler>(
		shm_holder.global_shared_memory.get_handler(fd));
	if (out_attr) {
		*out_attr = handler.attr;
	}
	if (out_name) {
		*out_name = handler.name.c_str();
	}
	if (type) {
		*type = handler.type;
	}
	return 0;
}

extern "C" uint64_t map_ptr_by_fd(uint32_t fd)
{
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	// Use a convenient way to represent a pointer
	return ((uint64_t)fd << 32) | 0xffffffff;
}

extern "C" uint64_t map_val(uint64_t map_ptr)
{
	int fd = (int)(map_ptr >> 32);
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	auto &handler = std::get<bpftime::bpf_map_handler>(
		shm_holder.global_shared_memory.get_handler(fd));
	auto size = handler.attr.key_size;
	std::vector<char> key(size);
	int res = handler.bpf_map_get_next_key(nullptr, key.data());
	if (res < 0) {
		errno = ENOENT;
		return 0;
	}
	return (uint64_t)handler.map_lookup_elem(key.data());
}
