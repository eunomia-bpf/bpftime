#ifndef _BPFTIME_SHM_INTERNAL
#define _BPFTIME_SHM_INTERNAL
#include "bpf_map/ringbuf_map.hpp"
#include "handler/epoll_handler.hpp"
#include "handler/map_handler.hpp"
#include "spdlog/spdlog.h"
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <functional>
#include <boost/interprocess/containers/set.hpp>
#include <common/bpftime_config.hpp>
#include <handler/handler_manager.hpp>
#include <optional>
#include <variant>
namespace bpftime
{

using syscall_pid_set_allocator = boost::interprocess::allocator<
	int, boost::interprocess::managed_shared_memory::segment_manager>;
using syscall_pid_set =
	boost::interprocess::set<int, std::less<int>, syscall_pid_set_allocator>;
// global bpftime share memory
class bpftime_shm {
	// shared memory segment
	boost::interprocess::managed_shared_memory segment;

	// manage the bpf fds in the shared memory
	bpftime::handler_manager *manager = nullptr;

	// A set to record whether a process was setted up with syscall tracer

	syscall_pid_set *syscall_installed_pids = nullptr;
	// Configuration for the agent. e.g, which helpers are enabled
	struct bpftime::agent_config *agent_config = nullptr;

    public:
	// Get the configuration object
	struct agent_config &get_agent_config()
	{
		return *agent_config;
	}
	// Check whether a certain pid was already equipped with syscall tracer
	// Using a set stored in the shared memory
	bool check_syscall_trace_setup(int pid);
	// Set whether a certain pid was already equipped with syscall tracer
	// Using a set stored in the shared memory
	void set_syscall_trace_setup(int pid, bool whether);

	const handler_variant &get_handler(int fd) const
	{
		return manager->get_handler(fd);
	}
	bool is_epoll_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			spdlog::error("Invalid fd: {}", fd);
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::epoll_handler>(handler);
	}

	bool is_map_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_map_handler>(
			handler);
	}
	bool is_ringbuf_map_fd(int fd) const
	{
		if (!is_map_fd(fd))
			return false;
		auto &map_impl =
			std::get<bpf_map_handler>(manager->get_handler(fd));
		return map_impl.type == map_impl.BPF_MAP_TYPE_RINGBUF;
	}
	std::optional<ringbuf_map_impl *> try_get_ringbuf_map_impl(int fd) const
	{
		if (!is_ringbuf_map_fd(fd)) {
			spdlog::error("Expected fd {} to be an ringbuf map fd",
				      fd);
			return {};
		}
		auto &map_handler =
			std::get<bpf_map_handler>(manager->get_handler(fd));
		return map_handler.try_get_ringbuf_map_impl();
	}
	bool is_prog_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_prog_handler>(
			handler);
	}

	bool is_perf_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_perf_event_handler>(
			handler);
	}

	int open_fake_fd()
	{
		return open("/dev/null", O_RDONLY);
	}

	// handle bpf commands to load a bpf program
	int add_bpf_prog(const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
	{
		int fd = open_fake_fd();
		manager->set_handler(
			fd,
			bpftime::bpf_prog_handler(segment, insn, insn_cnt,
						  prog_name, prog_type),
			segment);
		return fd;
	}

	// add a bpf link fd
	int add_bpf_link(int prog_fd, int target_fd)
	{
		int fd = open_fake_fd();
		if (!manager->is_allocated(target_fd) || !is_prog_fd(prog_fd)) {
			return -1;
		}
		manager->set_handler(
			fd,
			bpftime::bpf_link_handler{ (uint32_t)prog_fd,
						   (uint32_t)target_fd },
			segment);
		return fd;
	}

	// create a bpf map fd
	int add_bpf_map(const char *name, bpftime::bpf_map_attr attr)
	{
		int fd = open_fake_fd();
		if (!manager) {
			return -1;
		}
#ifdef ENABLE_BPFTIME_VERIFIER
		auto helpers = verifier::get_map_descriptors();
		helpers[fd] = verifier::BpftimeMapDescriptor{
			.original_fd = fd,
			.type = static_cast<uint32_t>(attr.type),
			.key_size = attr.key_size,
			.value_size = attr.value_size,
			.max_entries = attr.max_ents,
			.inner_map_fd = static_cast<unsigned int>(-1)
		};
		verifier::set_map_descriptors(helpers);
#endif
		manager->set_handler(
			fd, bpftime::bpf_map_handler(name, segment, attr),
			segment);
		return fd;
	}
	uint32_t bpf_map_value_size(int fd) const;
	const void *bpf_map_lookup_elem(int fd, const void *key) const;

	long bpf_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags) const;

	long bpf_delete_elem(int fd, const void *key) const;

	int bpf_map_get_next_key(int fd, const void *key, void *next_key) const;

	// create an uprobe fd
	int add_uprobe(int pid, const char *name, uint64_t offset,
		       bool retprobe, size_t ref_ctr_off);
	int add_tracepoint(int pid, int32_t tracepoint_id);
	int attach_perf_to_bpf(int perf_fd, int bpf_fd);
	int attach_enable(int fd) const;
	int add_ringbuf_to_epoll(int ringbuf_fd, int epoll_fd);
	int epoll_create();
	// remove a fake fd from the manager.
	// The fake fd should be closed by the caller.
	void close_fd(int fd)
	{
		if (manager) {
			manager->clear_fd_at(fd, segment);
		}
	}

	bool is_exist_fake_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		return manager->is_allocated(fd);
	}

	bpftime_shm()
	{
		spdlog::info("global_shm_open_type {} for {}",
			     (int)global_shm_open_type,
			     bpftime::get_global_shm_name());
		if (global_shm_open_type == shm_open_type::SHM_CLIENT) {
			// open the shm
			segment = boost::interprocess::managed_shared_memory(
				boost::interprocess::open_only,
				bpftime::get_global_shm_name());
			manager =
				segment.find<bpftime::handler_manager>(
					       bpftime::DEFAULT_GLOBAL_HANDLER_NAME)
					.first;
			syscall_installed_pids =
				segment.find<syscall_pid_set>(
					       DEFAULT_SYSCALL_PID_SET_NAME)
					.first;
			agent_config =
				segment.find<struct agent_config>(
					       bpftime::DEFAULT_AGENT_CONFIG_NAME)
					.first;
		} else if (global_shm_open_type == shm_open_type::SHM_SERVER) {
			boost::interprocess::shared_memory_object::remove(
				bpftime::get_global_shm_name());
			// create the shm
			segment = boost::interprocess::managed_shared_memory(
				boost::interprocess::create_only,
				bpftime::get_global_shm_name(), 1 << 20);
			manager = segment.construct<bpftime::handler_manager>(
				bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
			syscall_installed_pids =
				segment.construct<syscall_pid_set>(
					bpftime::DEFAULT_SYSCALL_PID_SET_NAME)(
					std::less<int>(),
					syscall_pid_set_allocator(
						segment.get_segment_manager()));
			agent_config = segment.construct<struct agent_config>(
				bpftime::DEFAULT_AGENT_CONFIG_NAME)();
		} else if (global_shm_open_type ==
			   shm_open_type::SHM_NO_CREATE) {
			// not create any shm
			return;
		}
	}

	const handler_manager *get_manager() const
	{
		return manager;
	}
};

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
extern bpftime_shm_holder shm_holder;
} // namespace bpftime
#endif
