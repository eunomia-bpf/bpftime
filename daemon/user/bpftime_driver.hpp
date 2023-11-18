/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_DRIVER_HPP
#define BPFTIME_DRIVER_HPP

#include "daemon_config.hpp"
#include <ebpf-vm.h>
#include <map>

extern "C" {
struct bpf_prog_info;
struct bpf_tracer_bpf;
}

namespace bpftime
{

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
	struct bpf_tracer_bpf *object = NULL;

	int check_and_create_prog_related_maps(int fd,
					       const bpf_prog_info *info);

    public:
	// create a bpf prog in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_progs_create_server(int kernel_id, int server_pid);

	// create a bpf map in the global shared memory
	//
	// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then
	// the function will allocate a new perf event fd.
	int bpftime_maps_create_server(int kernel_id);

	int bpftime_attach_perf_to_bpf_server(int server_pid, int perf_fd,
					      int kernel_bpf_id);
	int bpftime_attach_perf_to_bpf_fd_server(int server_pid, int perf_fd,
						 int bpf_prog_fd);

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

	// load the btf into kernel
	int bpftime_btf_load_server(int server_pid, int fd);

	void bpftime_close_server(int server_pid, int fd);

	bpftime_driver(struct daemon_config cfg, struct bpf_tracer_bpf *obj);
	~bpftime_driver();
};

} // namespace bpftime

#endif // BPFTIME_DRIVER_HPP
