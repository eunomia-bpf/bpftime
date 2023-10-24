#include "bpftime_driver.hpp"
#include "ebpf-vm.h"
#include <spdlog/spdlog.h>
#include "bpftime_shm.hpp"

using namespace bpftime;
using namespace std;

int bpftime_driver::find_minimal_unused_id()
{
	int id = 0;
	while (1) {
		bool find = false;
		for (auto [pid_fd, idx] : pid_fd_to_id_map) {
			if (id == idx) {
				find = true;
			}
		}
		if (find) {
			id++;
		} else {
			break;
		}
	}
	return id;
}

int bpftime_driver::bpftime_link_create_server(int server_pid, int fd,
					       int prog_fd, int target_fd)
{
	int id = find_minimal_unused_id();
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		spdlog::error("fd {} for pid {} not exists", fd, server_pid);
		return -1;
	}
	int prog_id = check_and_get_pid_fd(server_pid, prog_fd);
	if (prog_id < 0) {
		spdlog::error("prog fd {} for pid {} not exists", prog_fd,
			      server_pid);
		return -1;
	}

	int res = bpftime_link_create(id, prog_id, fd_id);
    if (res < 0) {
        spdlog::error("Failed to create link");
        return -1;
    }
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	return id;
}

int bpftime_driver::bpftime_progs_create_server(int server_pid, int fd,
						const ebpf_inst *insn,
						size_t insn_cnt,
						const char *prog_name,
						int prog_type)
{
	int id = find_minimal_unused_id();
    int res = bpftime_progs_create(id, insn, insn_cnt, prog_name, prog_type);
    if (res < 0) {
        spdlog::error("Failed to create prog");
        return -1;
    }
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	return id;
}

int bpftime_driver::bpftime_maps_create_server(int server_pid, int fd,
					       const char *name,
					       bpftime::bpf_map_attr attr)
{
    int id = find_minimal_unused_id();
    int res = bpftime_maps_create(id, name, attr);
    if (res < 0) {
        spdlog::error("Failed to create map");
        return -1;
    }
    pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
    return id;
}

int bpftime_driver::bpftime_attach_perf_to_bpf_server(int server_pid,
						      int perf_fd, int bpf_fd)
{
    int perf_id = check_and_get_pid_fd(server_pid, perf_fd);
    if (perf_id < 0) {
        spdlog::error("perf fd {} for pid {} not exists", perf_fd,
                  server_pid);
        return -1;
    }
    int bpf_id = check_and_get_pid_fd(server_pid, bpf_fd);
    if (bpf_id < 0) {
        spdlog::error("bpf fd {} for pid {} not exists", bpf_fd,
                  server_pid);
        return -1;
    }
    int res = bpftime_attach_perf_to_bpf(perf_id, bpf_id);
    if (res < 0) {
        spdlog::error("Failed to attach perf to bpf");
        return -1;
    }
    return 0;
}

int bpftime_driver::bpftime_uprobe_create_server(int server_pid, int fd,
						 int target_pid,
						 const char *name,
						 uint64_t offset, bool retprobe,
						 size_t ref_ctr_off)
{
    int id = find_minimal_unused_id();
    int fd_id = check_and_get_pid_fd(server_pid, fd);
    if (fd_id < 0) {
        spdlog::error("fd {} for pid {} not exists", fd, server_pid);
        return -1;
    }
    int res = bpftime_uprobe_create(id, target_pid, name, offset, retprobe,
                    ref_ctr_off);
    if (res < 0) {
        spdlog::error("Failed to create uprobe");
        return -1;
    }
    pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
    return id;
}

bpftime_driver::bpftime_driver(daemon_config cfg)
{
	config = cfg;
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
}