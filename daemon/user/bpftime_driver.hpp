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
	std::map<uint64_t, int> pid_fd_to_id_map;

    public:
	// create a bpf link in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_link_create(int server_pid, int fd, int prog_fd,
				int target_fd);

	// create a bpf prog in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_progs_create(int server_pid, int fd, const ebpf_inst *insn,
				 size_t insn_cnt, const char *prog_name,
				 int prog_type);

	// create a bpf map in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_maps_create(int fd, const char *name,
				bpftime::bpf_map_attr attr);

	// create uprobe in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_uprobe_create(int server_pid, int fd, int target_pid,
				  const char *name, uint64_t offset,
				  bool retprobe, size_t ref_ctr_off);
	// create tracepoint in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_tracepoint_create(int server_pid, int fd, int pid,
				      int32_t tp_id);
	// enable the perf event
	int bpftime_perf_event_enable(int server_pid, int fd);
	// disable the perf event
	int bpftime_perf_event_disable(int server_pid, int fd);

	void bpftime_close(int server_pid, int fd);

	bpftime_driver(struct daemon_config cfg);
	~bpftime_driver();
};

} // namespace bpftime

#endif // BPFTIME_DRIVER_HPP
