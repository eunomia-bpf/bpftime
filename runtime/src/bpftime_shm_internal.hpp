#ifndef _BPFTIME_SHM_INTERNAL
#define _BPFTIME_SHM_INTERNAL
#include "bpf_map/array_map.hpp"
#include "bpf_map/ringbuf_map.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstddef>
#include <functional>
#include <boost/interprocess/containers/set.hpp>
#include <common/bpftime_config.hpp>
#include <handler/handler_manager.hpp>
#include <optional>
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

	const handler_variant &get_handler(int fd) const;
	bool is_epoll_fd(int fd) const;

	bool is_map_fd(int fd) const;
	bool is_ringbuf_map_fd(int fd) const;
	bool is_array_map_fd(int fd) const;
	bool is_perf_event_handler_fd(int fd) const;
	bool is_software_perf_event_handler_fd(int fd) const;

	std::optional<ringbuf_map_impl *>
	try_get_ringbuf_map_impl(int fd) const;

	std::optional<array_map_impl *> try_get_array_map_impl(int fd) const;
	bool is_prog_fd(int fd) const;

	bool is_perf_fd(int fd) const;

	int open_fake_fd();

	// handle bpf commands to load a bpf program
	int add_bpf_prog(const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type);

	// add a bpf link fd
	int add_bpf_link(int prog_fd, int target_fd);

	// create a bpf map fd
	int add_bpf_map(const char *name, bpftime::bpf_map_attr attr);
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
	int add_software_perf_event(int cpu, int32_t sample_type,
				    int64_t config);
	int attach_perf_to_bpf(int perf_fd, int bpf_fd);
	int attach_enable(int fd) const;
	int add_ringbuf_to_epoll(int ringbuf_fd, int epoll_fd,
				 epoll_data_t extra_data);
	int add_software_perf_event_to_epoll(int swpe_fd, int epoll_fd,
					     epoll_data_t extra_data);

	int epoll_create();
	// remove a fake fd from the manager.
	// The fake fd should be closed by the caller.
	void close_fd(int fd);
	bool is_exist_fake_fd(int fd) const;
	bpftime_shm();
	const handler_manager *get_manager() const;
	std::optional<void *>
	get_software_perf_event_raw_buffer(int fd, size_t buffer_sz) const;
};

// memory region for maps and prog info
// Use union so that the destructor of bpftime_shm won't be called automatically
union bpftime_shm_holder {
	bpftime_shm global_shared_memory;
	bpftime_shm_holder()
	{
		
	}
	~bpftime_shm_holder()
	{
	}
};
extern bpftime_shm_holder shm_holder;

} // namespace bpftime
extern "C" void bpftime_initialize_global_shm();
extern "C" void bpftime_destroy_global_shm();
#endif
