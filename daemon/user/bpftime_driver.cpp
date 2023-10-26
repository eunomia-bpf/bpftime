#include "bpftime_driver.hpp"
#include "ebpf-vm.h"
#include <spdlog/spdlog.h>
#include "bpftime_shm.hpp"

using namespace bpftime;
using namespace std;

int bpftime_driver::find_minimal_unused_id()
{
	int id = bpftime_find_minimal_unused_fd();
	spdlog::debug("find minimal unused id {}", id);
	return id;
}

int bpftime_driver::bpftime_link_create_server(int server_pid, int fd,
					       int prog_fd, int target_fd)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		spdlog::error("Failed to find minimal unused id");
		return -1;
	}
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
		spdlog::error("Failed to create link for id {}", id);
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	spdlog::info("create link {} for pid {} fd {}", id, server_pid, fd);
	return id;
}

int bpftime_driver::bpftime_progs_create_server(int server_pid, int fd,
						const ebpf_inst *insn,
						size_t insn_cnt,
						const char *prog_name,
						int prog_type)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		spdlog::error("Failed to find minimal unused id");
		return -1;
	}
	int res =
		bpftime_progs_create(id, insn, insn_cnt, prog_name, prog_type);
	if (res < 0) {
		spdlog::error("Failed to create prog for id {}", id);
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	spdlog::info("create prog {} for pid {} fd {}", id, server_pid, fd);
	return id;
}

int bpftime_driver::bpftime_maps_create_server(int server_pid, int fd,
					       const char *name,
					       bpftime::bpf_map_attr attr)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		spdlog::error("Failed to find minimal unused id");
		return -1;
	}
	int res = bpftime_maps_create(id, name, attr);
	if (res < 0) {
		spdlog::error("Failed to create map for id {}", id);
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	spdlog::info("create map {} for pid {} fd {}", id, server_pid, fd);
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
	spdlog::info("attach perf {} to bpf {}, for pid {}", perf_id, bpf_id,
		     server_pid);
	return 0;
}

int bpftime_driver::bpftime_uprobe_create_server(int server_pid, int fd,
						 int target_pid,
						 const char *name,
						 uint64_t offset, bool retprobe,
						 size_t ref_ctr_off)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		spdlog::error("Failed to find minimal unused id");
		return -1;
	}
	int res = bpftime_uprobe_create(id, target_pid, name, offset, retprobe,
					ref_ctr_off);
	if (res < 0) {
		spdlog::error("Failed to create uprobe");
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	spdlog::info("create uprobe {} for pid {} fd {}", id, server_pid, fd);
	return id;
}

// enable the perf event
int bpftime_driver::bpftime_perf_event_enable_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		spdlog::error("fd {} for pid {} not exists", fd, server_pid);
		return -1;
	}
	int res = bpftime_perf_event_enable(fd_id);
	if (res < 0) {
		spdlog::error("Failed to enable perf event");
		return -1;
	}
	spdlog::info("enable perf event {} for pid {} fd {}", fd_id, server_pid,
		     fd);
	return 0;
}
// disable the perf event
int bpftime_driver::bpftime_perf_event_disable_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		spdlog::error("fd {} for pid {} not exists", fd, server_pid);
		return -1;
	}
	int res = bpftime_perf_event_disable(fd_id);
	if (res < 0) {
		spdlog::error("Failed to disable perf event");
		return -1;
	}
	spdlog::info("disable perf event {} for pid {} fd {}", fd_id,
		     server_pid, fd);
	return 0;
}

void bpftime_driver::bpftime_close_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		spdlog::error("fd {} for pid {} not exists", fd, server_pid);
		return;
	}
	pid_fd_to_id_map.erase(get_pid_fd_key(server_pid, fd));
	bpftime_close(fd_id);
	spdlog::info("close id {} for pid {} fd {}", fd_id, server_pid, fd);
}

int bpftime_driver::bpftime_btf_load_server(int server_pid, int fd)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		spdlog::error("Failed to find minimal unused id");
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	// TODO: handle kernel BTF in our system
	spdlog::info("create btf {} for pid {} fd {}", id, server_pid, fd);
	return 0;
}

bpftime_driver::bpftime_driver(daemon_config cfg)
{
	config = cfg;
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
}

bpftime_driver::~bpftime_driver()
{
	bpftime_destroy_global_shm();
}
