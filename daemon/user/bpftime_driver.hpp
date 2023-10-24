#ifndef BPFTIME_DRIVER_HPP
#define BPFTIME_DRIVER_HPP

#include "daemon_config.hpp"
#include <string>
#include <ebpf-vm.h>
#include <map>

namespace bpftime
{

// bpf map attribute
struct bpf_map_attr {
	int type = 0;
	uint32_t key_size = 0;
	uint32_t value_size = 0;
	uint32_t max_ents = 0;
	uint64_t flags = 0;
	uint32_t ifindex = 0;
	uint32_t btf_vmlinux_value_type_id = 0;
	uint32_t btf_id = 0;
	uint32_t btf_key_type_id = 0;
	uint32_t btf_value_type_id = 0;
	uint64_t map_extra = 0;
};

// Use commands to interact with the bpftime agent and shm maps
class bpftime_driver {
	int find_minimal_unused_id();
	inline uint64_t get_pid_fd_key(int pid, int fd)
	{
		return ((uint64_t)pid << 32) | (uint64_t)fd;
	}
	inline int check_and_get_pid_fd(int pid, int fd)
	{
		auto it = pid_fd_to_id_map.find(get_pid_fd_key(pid, fd));
		if (it == pid_fd_to_id_map.end()) {
			return -1;
		}
		return it->second;
	}
	std::map<uint64_t, int> pid_fd_to_id_map;
	daemon_config config;

    public:
	// create a bpf link in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_link_create_server(int server_pid, int fd, int prog_fd,
				int target_fd);

	// create a bpf prog in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_progs_create_server(int server_pid, int fd, const ebpf_inst *insn,
				 size_t insn_cnt, const char *prog_name,
				 int prog_type);

	// create a bpf map in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_maps_create_server(int server_pid, int fd, const char *name,
				bpftime::bpf_map_attr attr);

	int bpftime_attach_perf_to_bpf_server(int server_pid, int perf_fd, int bpf_fd);

	// create uprobe in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_uprobe_create_server(int server_pid, int fd, int target_pid,
				  const char *name, uint64_t offset,
				  bool retprobe, size_t ref_ctr_off);
	// create tracepoint in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_tracepoint_create_server(int server_pid, int fd, int pid,
				      int32_t tp_id);
	// enable the perf event
	int bpftime_perf_event_enable_server(int server_pid, int fd);
	// disable the perf event
	int bpftime_perf_event_disable_server(int server_pid, int fd);

	void bpftime_close_server(int server_pid, int fd);

	bpftime_driver(struct daemon_config cfg);
	~bpftime_driver() = default;
};

} // namespace bpftime

#endif // BPFTIME_DRIVER_HPP
