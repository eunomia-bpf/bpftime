/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_driver.hpp"
#include <linux/bpf.h>
#include <bpf/bpf.h>
#include "ebpf_inst.h"
#include <spdlog/spdlog.h>
#include "bpftime_shm.hpp"
#include <vector>
#include <cassert>
#include <linux/perf_event.h>
#include "../bpf_tracer_event.h"
#include "bpf_tracer.skel.h"

using namespace bpftime;
using namespace std;

int bpftime_driver::find_minimal_unused_id()
{
	int id = bpftime_find_minimal_unused_fd();
	SPDLOG_DEBUG("find minimal unused id {}", id);
	return id;
}

static int get_kernel_bpf_prog_insns(int fd, const bpf_prog_info *info,
				     std::vector<ebpf_inst> &insns)
{
	insns.resize(info->xlated_prog_len / sizeof(ebpf_inst));
	bpf_prog_info new_info = {};
	uint32_t info_len = sizeof(bpf_prog_info);

	new_info.xlated_prog_len = info->xlated_prog_len;
	new_info.xlated_prog_insns =
		(unsigned long long)(uintptr_t)insns.data();

	int res = bpf_obj_get_info_by_fd(fd, &new_info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR("Failed to get prog info for fd {}", fd);
		return -1;
	}
	return 0;
}

struct bpf_insn_data insn_data;

static int get_bpf_obj_id_from_pid_fd(bpf_tracer_bpf *obj, int pid, int fd)
{
	unsigned long long key = MAKE_PFD(pid, fd);
	int map_id = -1;
	struct bpf_fd_data data = {};

	int res = bpf_map__lookup_elem(obj->maps.bpf_fd_map, &key, sizeof(key),
				       &data, sizeof(data), 0);
	if (res < 0) {
		SPDLOG_ERROR("Failed to lookup bpf fd map for pid {} fd {}",
			      pid, fd);
		return -1;
	}
	// if (data.type != BPF_FD_TYPE_MAP) {
	// 	SPDLOG_ERROR("Invalid bpf fd type {} for pid {} fd {}",
	// 		      (int)data.type, pid, fd);
	// }
	map_id = data.kernel_id;
	return map_id;
}

static int relocate_bpf_prog_insns(std::vector<ebpf_inst> &insns,
				   bpf_tracer_bpf *obj, int id, int pid)
{
	// relocate the bpf program
	unsigned long long key_id = id;
	assert(obj);
	assert(obj->maps.bpf_prog_insns_map);
	int res = bpf_map__lookup_elem(obj->maps.bpf_prog_insns_map, &key_id,
				       sizeof(key_id), &insn_data,
				       sizeof(insn_data), 0);
	if (res < 0) {
		SPDLOG_ERROR("Failed to lookup bpf prog insns for id {}", id);
		return -1;
	}
	SPDLOG_DEBUG("relocate bpf prog insns for id {}, cnt {}", id,
		      insn_data.code_len);
	// resize the insns
	insns.resize(insn_data.code_len);
	const ebpf_inst *orignal_insns = (const ebpf_inst *)insn_data.code;
	insns.assign(orignal_insns, orignal_insns + insn_data.code_len);
	for (size_t i = 0; i < insn_data.code_len; i++) {
		const struct ebpf_inst inst = orignal_insns[i];
		bool store = false;

		switch (inst.code) {
		case EBPF_OP_LDDW:
			if (inst.src_reg == 1 || inst.src_reg == 2) {
				int map_id = get_bpf_obj_id_from_pid_fd(
					obj, pid, inst.imm);
				if (map_id < 0) {
					return -1;
				}
				SPDLOG_DEBUG(
					"relocate bpf prog insns for id {} in {}, "
					"lddw imm {} to map id {}",
					id, i, inst.imm, map_id);
				insns[i].imm = map_id;
			}
			break;
		case EBPF_OP_CALL:
			SPDLOG_DEBUG(
				"relocate bpf prog insns for id {} in {}, call imm {}",
				id, i, inst.imm);
			break;
		default:
			break;
		}
	}
	return 0;
}

int bpftime_driver::check_and_create_prog_related_maps(
	int fd, const bpf_prog_info *info)
{
	std::vector<unsigned int> map_ids;
	uint32_t info_len = sizeof(bpf_prog_info);
	bpf_prog_info new_info = {};

	SPDLOG_INFO("find {} related maps for prog fd {}", info->nr_map_ids,
		     fd);
	map_ids.resize(info->nr_map_ids);
	new_info.nr_map_ids = info->nr_map_ids;
	new_info.map_ids = (unsigned long long)(uintptr_t)map_ids.data();
	int res = bpf_obj_get_info_by_fd(fd, &new_info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR(
			"Failed to get prog info for fd {} to find related maps",
			fd);
		return -1;
	}
	assert(info->nr_map_ids == new_info.nr_map_ids);
	for (unsigned int i = 0; i < info->nr_map_ids; i++) {
		int map_id = map_ids[i];
		if (bpftime_is_map_fd(map_id)) {
			// check whether the map is exist
			SPDLOG_INFO("map {} already exists", map_id);
			continue;
		}
		int res = bpftime_maps_create_server(map_id);
		if (res < 0) {
			SPDLOG_ERROR("Failed to create map for id {}", map_id);
			return -1;
		}
	}
	return 0;
}

int bpftime_driver::bpftime_progs_create_server(int kernel_id, int server_pid)
{
	int fd = bpf_prog_get_fd_by_id(kernel_id);
	if (fd < 0) {
		SPDLOG_ERROR("Failed to get prog fd by prog id {}, err={}",
			      kernel_id, errno);
		return -1;
	}
	SPDLOG_DEBUG("get prog fd {} for id {}", fd, kernel_id);
	bpf_prog_info info = {};
	uint32_t info_len = sizeof(info);
	int res = bpf_obj_get_info_by_fd(fd, &info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR("Failed to get prog info for id {}", kernel_id);
		return -1;
	}
	if (bpftime_is_prog_fd(kernel_id)) {
		// check whether the prog is exist
		SPDLOG_INFO("prog {} already exists in shm", kernel_id);
		return 0;
	}
	std::vector<ebpf_inst> buffer;
	res = relocate_bpf_prog_insns(buffer, object, kernel_id, server_pid);
	if (res < 0) {
		SPDLOG_ERROR("Failed to relocate prog insns for id {}",
			      kernel_id);
		return -1;
	}
	res = bpftime_progs_create(kernel_id, buffer.data(), buffer.size(),
				   info.name, info.type);
	if (res < 0) {
		SPDLOG_ERROR("Failed to create prog for id {}", kernel_id);
		return -1;
	}
	SPDLOG_INFO("create prog {}, fd {} in shm success", kernel_id, fd);
	// check and created related maps
	res = check_and_create_prog_related_maps(fd, &info);
	if (res < 0) {
		SPDLOG_ERROR("Failed to create related maps for prog {}",
			      kernel_id);
		return -1;
	}
	close(fd);
	return kernel_id;
}

int bpftime_driver::bpftime_maps_create_server(int kernel_id)
{
	int map_fd = bpf_map_get_fd_by_id(kernel_id);
	if (map_fd < 0) {
		SPDLOG_ERROR("Failed to get map fd for id {}", kernel_id);
		return -1;
	}
	bpf_map_info info = {};
	uint32_t info_len = sizeof(info);
	int res = bpf_obj_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR("Failed to get map info for id {}", kernel_id);
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
		SPDLOG_INFO("map {} already exists", kernel_id);
		return 0;
	}

	res = bpftime_maps_create(kernel_id, info.name, attr);
	if (res < 0) {
		SPDLOG_ERROR("Failed to create map for id {}", kernel_id);
		return -1;
	}
	SPDLOG_INFO("create map in kernel id {}", kernel_id);
	return kernel_id;
}

int bpftime_driver::bpftime_attach_perf_to_bpf_fd_server(int server_pid,
							 int perf_fd,
							 int bpf_prog_fd)
{
	int prog_id =
		get_bpf_obj_id_from_pid_fd(object, server_pid, bpf_prog_fd);
	if (prog_id < 0) {
		SPDLOG_ERROR(
			"Failed to lookup bpf prog id from bpf prog fd {}",
			bpf_prog_fd);
		return -1;
	}
	return bpftime_attach_perf_to_bpf_server(server_pid, perf_fd, prog_id);
}

int bpftime_driver::bpftime_attach_perf_to_bpf_server(int server_pid,
						      int perf_fd,
						      int kernel_bpf_id)
{
	int perf_id = check_and_get_pid_fd(server_pid, perf_fd);
	if (perf_id < 0) {
		SPDLOG_ERROR("perf fd {} for pid {} not exists", perf_fd,
			      server_pid);
		return -1;
	}
	if (bpftime_is_prog_fd(kernel_bpf_id)) {
		SPDLOG_INFO("bpf {} already exists in shm", kernel_bpf_id);
	} else {
		int res =
			bpftime_progs_create_server(kernel_bpf_id, server_pid);
		if (res < 0) {
			SPDLOG_ERROR(
				"Failed to create bpf program (userspace side) for kernel program id {}",
				kernel_bpf_id);
			return -1;
		}
	}

	int res = bpftime_attach_perf_to_bpf(perf_id, kernel_bpf_id);
	if (res < 0) {
		SPDLOG_ERROR("Failed to attach perf to bpf");
		return -1;
	}
	SPDLOG_INFO("attach perf {} to bpf {}, for pid {}", perf_id,
		     kernel_bpf_id, server_pid);
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
		SPDLOG_ERROR("Failed to find minimal unused id");
		return -1;
	}
	int res = bpftime_uprobe_create(id, target_pid, name, offset, retprobe,
					ref_ctr_off);
	if (res < 0) {
		SPDLOG_ERROR("Failed to create uprobe");
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	SPDLOG_INFO("create uprobe {} for pid {} fd {}", id, server_pid, fd);
	return id;
}

// enable the perf event
int bpftime_driver::bpftime_perf_event_enable_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		spdlog::warn("Unrecorded uprobe fd: {} for pid {}", fd,
			     server_pid);
		return 0;
	}
	int res = bpftime_perf_event_enable(fd_id);
	if (res < 0) {
		SPDLOG_ERROR("Failed to enable perf event");
		return -1;
	}
	SPDLOG_INFO("enable perf event {} for pid {} fd {}", fd_id, server_pid,
		     fd);
	return 0;
}

// disable the perf event
int bpftime_driver::bpftime_perf_event_disable_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		SPDLOG_ERROR("fd {} for pid {} not exists", fd, server_pid);
		return -1;
	}
	int res = bpftime_perf_event_disable(fd_id);
	if (res < 0) {
		SPDLOG_ERROR("Failed to disable perf event");
		return -1;
	}
	SPDLOG_INFO("disable perf event {} for pid {} fd {}", fd_id,
		     server_pid, fd);
	return 0;
}

void bpftime_driver::bpftime_close_server(int server_pid, int fd)
{
	int fd_id = check_and_get_pid_fd(server_pid, fd);
	if (fd_id < 0) {
		SPDLOG_INFO("fd {} for pid {} not exists", fd, server_pid);
		return;
	}
	pid_fd_to_id_map.erase(get_pid_fd_key(server_pid, fd));
	bpftime_close(fd_id);
	SPDLOG_INFO("close id {} for pid {} fd {}", fd_id, server_pid, fd);
}

int bpftime_driver::bpftime_btf_load_server(int server_pid, int fd)
{
	int id = find_minimal_unused_id();
	if (id < 0) {
		SPDLOG_ERROR("Failed to find minimal unused id");
		return -1;
	}
	pid_fd_to_id_map[get_pid_fd_key(server_pid, fd)] = id;
	// TODO: handle kernel BTF in our system
	SPDLOG_INFO("create btf {} for pid {} fd {}", id, server_pid, fd);
	return 0;
}

bpftime_driver::bpftime_driver(daemon_config cfg, struct bpf_tracer_bpf *obj)
{
	config = cfg;
	object = obj;
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	auto config = get_agent_config_from_env();
	bpftime_set_agent_config(config);
}

bpftime_driver::~bpftime_driver()
{
	bpftime_destroy_global_shm();
}
