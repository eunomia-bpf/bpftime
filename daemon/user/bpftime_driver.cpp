#include "bpftime_driver.hpp"
#include "ebpf-vm.h"
#include <spdlog/spdlog.h>
#include "bpftime_shm.hpp"
#include <bpf/bpf.h>
#include <vector>
#include <cassert>

using namespace bpftime;
using namespace std;

int bpftime_driver::find_minimal_unused_id()
{
	int id = bpftime_find_minimal_unused_fd();
	spdlog::debug("find minimal unused id {}", id);
	return id;
}

static int get_kernel_bpf_prog_insns(int fd, bpf_prog_info *info,
				     std::vector<ebpf_inst> &insns)
{
	insns.resize(info->xlated_prog_len);
	uint32_t info_len = sizeof(bpf_prog_info);
	info->xlated_prog_insns = (unsigned long long)(uintptr_t)insns.data();
	int res = bpf_obj_get_info_by_fd(fd, info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get prog info for fd {}", fd);
		return -1;
	}
	return 0;
}

int bpftime_driver::check_and_create_prog_related_maps(int fd, bpf_prog_info* info) {
	std::vector<int> map_ids;
	map_ids.resize(info->nr_map_ids);
	uint32_t info_len = sizeof(bpf_prog_info);
	info->map_ids = (unsigned long long)(uintptr_t)map_ids.data();
	int res = bpf_obj_get_info_by_fd(fd, info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get prog info for fd {} to find related maps", fd);
		return -1;
	}
	for (int i = 0; i < info->nr_map_ids; i++) {
		int map_id = map_ids[i];
		if (bpftime_is_map_fd(map_id)) {
			// check whether the map is exist
			spdlog::info("map {} already exists", map_id);
			continue;
		}
		int res = bpftime_maps_create_server(map_id);
		if (res < 0) {
			spdlog::error("Failed to create map for id {}", map_id);
			return -1;
		}
	}
	return 0;
}

int bpftime_driver::bpftime_progs_create_server(int kernel_id)
{
	int fd = bpf_prog_get_fd_by_id(kernel_id);
	if (fd < 0) {
		spdlog::error("Failed to get prog fd for id {}", kernel_id);
		return -1;
	}
	bpf_prog_info info = {};
	uint32_t info_len = sizeof(info);
	int res = bpf_obj_get_info_by_fd(fd, &info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get prog info for id {}", kernel_id);
		return -1;
	}
	if (bpftime_is_prog_fd(kernel_id)) {
		// check whether the prog is exist
		spdlog::info("prog {} already exists in shm", kernel_id);
		return 0;
	}
	std::vector<ebpf_inst> buffer;
	res = get_kernel_bpf_prog_insns(fd, &info, buffer);
	if (res < 0) {
		spdlog::error("Failed to get prog insns for id {}", kernel_id);
		return -1;
	}
	res = bpftime_progs_create(kernel_id, buffer.data(), buffer.size(),
				   info.name, info.type);
	if (res < 0) {
		spdlog::error("Failed to create prog for id {}", kernel_id);
		return -1;
	}
	spdlog::info("create prog {} in shm success", kernel_id);
	// check and created related maps
	res = check_and_create_prog_related_maps(fd, &info);
	if (res < 0) {
		spdlog::error("Failed to create related maps for prog {}", kernel_id);
		return -1;
	}
	return kernel_id;
}

int bpftime_driver::bpftime_maps_create_server(int kernel_id)
{
	int map_fd = bpf_map_get_fd_by_id(kernel_id);
	if (map_fd < 0) {
		spdlog::error("Failed to get map fd for id {}", kernel_id);
		return -1;
	}
	bpf_map_info info = {};
	uint32_t info_len = sizeof(info);
	int res = bpf_map_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get map info for id {}", kernel_id);
		return -1;
	}
	bpftime::bpf_map_attr attr;
	// convert type to kernel-user type
	attr.type = info.type + KERNEL_USER_MAP_OFFSET;
	attr.key_size = info.key_size;
	attr.value_size = info.value_size;
	attr.max_ents = info.max_entries;
	attr.flags = info.map_flags;
	attr.kernel_bpf_map_id = kernel_id;
	attr.btf_id = info.btf_id;
	attr.btf_key_type_id = info.btf_key_type_id;
	attr.btf_value_type_id = info.btf_value_type_id;
	attr.btf_vmlinux_value_type_id = info.btf_vmlinux_value_type_id;
	attr.ifindex = info.ifindex;

	if (bpftime_is_map_fd(kernel_id)) {
		// check whether the map is exist
		spdlog::info("map {} already exists", kernel_id);
		return 0;
	}

	res = bpftime_maps_create(kernel_id, info.name, attr);
	if (res < 0) {
		spdlog::error("Failed to create map for id {}", kernel_id);
		return -1;
	}
	spdlog::info("create map in kernel id {}", kernel_id);
	return kernel_id;
}

int bpftime_driver::bpftime_attach_perf_to_bpf_server(int server_pid,
						      int perf_fd, int bpf_fd,
						      int kernel_bpf_id)
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
		spdlog::warn("fd {} for pid {} not exists", fd, server_pid);
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
