/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_logger.hpp"
#include "bpftime_shm.hpp"
#include <cstdio>
#include <ebpf-vm.h>
#include "syscall_context.hpp"
#include "handler/map_handler.hpp"
#include <cstring>
#include <fcntl.h>
#if __linux__
#include "linux/perf_event.h"
#include <linux/bpf.h>
#include <sys/epoll.h>
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include <linux/filter.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstdlib>
#include "syscall_server_utils.hpp"
#include <optional>
#include <sys/mman.h>
#include <unistd.h>
#include <regex>

// In build option without libbpf, there might be no BPF_EXIT_INSN
#ifndef BPF_EXIT_INSN
#define BPF_EXIT_INSN()                                                        \
	((struct bpf_insn){ .code = BPF_JMP | BPF_EXIT,                        \
			    .dst_reg = 0,                                      \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = 0 })
#endif

#ifndef BPF_MOV64_IMM
#define BPF_MOV64_IMM(DST, IMM)                                                \
	((struct bpf_insn){ .code = BPF_ALU64 | BPF_MOV | BPF_K,               \
			    .dst_reg = DST,                                    \
			    .src_reg = 0,                                      \
			    .off = 0,                                          \
			    .imm = IMM })
#endif

using namespace bpftime;
#if __APPLE__
using namespace bpftime_epoll;
#endif

void syscall_context::load_config_from_env()
{
	const char *run_with_kernel_env = getenv("BPFTIME_RUN_WITH_KERNEL");
	if (run_with_kernel_env != nullptr) {
		run_with_kernel = true;
		SPDLOG_INFO("Using kernel eBPF runtime and maps");
	} else {
		run_with_kernel = false;
	}
	const char *not_load_pattern = getenv("BPFTIME_NOT_LOAD_PATTERN");
	if (not_load_pattern != nullptr) {
		SPDLOG_INFO("By pass kernel verifier pattern: {}",
			    not_load_pattern);
		by_pass_kernel_verifier_pattern = std::string(not_load_pattern);
	} else {
		by_pass_kernel_verifier_pattern.clear();
	}
}

syscall_context::syscall_context()
{
	init_original_functions();
	// FIXME: merge this into the runtime config
	load_config_from_env();
	auto runtime_config = bpftime::get_agent_config_from_env();
	pthread_spin_init(&this->mocked_file_lock, 0);
	SPDLOG_INFO("Init bpftime syscall mocking..");
	SPDLOG_INFO("The log will be written to: {}",
		    runtime_config.logger_output_path);
}

void syscall_context::try_startup()
{
	enable_mock = false;
	start_up();
	enable_mock = true;
}

int syscall_context::handle_close(int fd)
{
	if (!enable_mock)
		return orig_close_fn(fd);
	try_startup();
	{
		bpftime_lock_guard _guard(this->mocked_file_lock);
		if (auto itr = this->mocked_files.find(fd);
		    itr != this->mocked_files.end()) {
			SPDLOG_DEBUG("Removing mocked file fd {}", fd);
			this->mocked_files.erase(itr);
			return 0;
		}
	}
	bpftime_close(fd);
	return orig_close_fn(fd);
}

int syscall_context::handle_openat(int fd, const char *file, int oflag,
				   unsigned short mode)
{
	if (!enable_mock)
		return orig_openat_fn(fd, file, oflag, mode);
	try_startup();
	auto path = resolve_filename_and_fd_to_full_path(fd, file);
	if (!path) {
		SPDLOG_WARN("Failed to resolve fd={}/file=`{}`", fd, file);
		return orig_openat_fn(fd, file, oflag, mode);
	}
	if (auto mocker = create_mocked_file_based_on_full_path(*path);
	    mocker) {
		bpftime_lock_guard _guard(this->mocked_file_lock);
		char filename_buf[] = "/tmp/bpftime-mock.XXXXXX";
		int fake_fd = mkstemp(filename_buf);
		if (fake_fd < 0) {
			SPDLOG_WARN("Unable to create mock fd: {}", errno);
			return orig_open_fn(file, oflag, mode);
		}
		this->mocked_files.emplace(fake_fd, std::move(*mocker));
		SPDLOG_DEBUG("Created mocked file with fd {}", fake_fd);
		return fake_fd;
	}
	return orig_openat_fn(fd, file, oflag, mode);
}

int syscall_context::handle_open(const char *file, int oflag,
				 unsigned short mode)
{
	if (!enable_mock)
		return orig_open_fn(file, oflag, mode);
	try_startup();
	if (auto mocker = create_mocked_file_based_on_full_path(file); mocker) {
		bpftime_lock_guard _guard(this->mocked_file_lock);
		char filename_buf[] = "/tmp/bpftime-mock.XXXXXX";
		int fake_fd = mkstemp(filename_buf);
		if (fake_fd < 0) {
			SPDLOG_WARN("Unable to create mock fd: {}", errno);
			return orig_open_fn(file, oflag, mode);
		}
		this->mocked_files.emplace(fake_fd, std::move(*mocker));
		SPDLOG_DEBUG("Created mocked file with fd {}", fake_fd);
		return fake_fd;
	}
	return orig_open_fn(file, oflag, mode);
}

ssize_t syscall_context::handle_read(int fd, void *buf, size_t count)
{
	if (!enable_mock)
		return orig_read_fn(fd, buf, count);
	try_startup();
	{
		bpftime_lock_guard _guard(this->mocked_file_lock);
		if (auto itr = this->mocked_files.find(fd);
		    itr != this->mocked_files.end()) {
			SPDLOG_DEBUG("Mock read fd={}, buf={:x}, count={}", fd,
				     (uintptr_t)buf, count);
			auto &mock_file = itr->second;
			bpftime_lock_guard _access_guard(
				mock_file->access_lock);
			auto can_read_bytes =
				std::min(count, mock_file->buf.size() -
							mock_file->cursor);
			SPDLOG_DEBUG("Reading {} bytes", can_read_bytes);
			if (can_read_bytes == 0)
				return can_read_bytes;
			memcpy(buf, &mock_file->buf[mock_file->cursor],
			       can_read_bytes);
			mock_file->cursor += can_read_bytes;
			SPDLOG_DEBUG("Copied {} bytes", can_read_bytes);

			return can_read_bytes;
		}
	}
	return orig_read_fn(fd, buf, count);
}
int syscall_context::create_kernel_bpf_map(int map_fd)
{
	bpf_map_info info = {};
	uint32_t info_len = sizeof(info);
	int res = bpf_obj_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		SPDLOG_ERROR("Failed to get map info for id {}", info.id);
		return -1;
	}
	bpftime::bpf_map_attr attr;
	// convert type to kernel-user type
	attr.type = info.type + KERNEL_USER_MAP_OFFSET;
	attr.key_size = info.key_size;
	attr.value_size = info.value_size;
	attr.max_ents = info.max_entries;
	attr.flags = info.map_flags;
	attr.kernel_bpf_map_id = info.id;
	attr.btf_id = info.btf_id;
	attr.btf_key_type_id = info.btf_key_type_id;
	attr.btf_value_type_id = info.btf_value_type_id;
	attr.btf_vmlinux_value_type_id = info.btf_vmlinux_value_type_id;
	attr.ifindex = info.ifindex;

	if (bpftime_is_map_fd(info.id)) {
		// check whether the map is exist
		SPDLOG_INFO("map {} already exists", info.id);
		return 0;
	}

	res = bpftime_maps_create(info.id, info.name, attr);
	if (res < 0) {
		SPDLOG_ERROR("Failed to create map for id {}", info.id);
		return -1;
	}
	SPDLOG_INFO("create map in kernel id {}", info.id);
	return map_fd;
}

int syscall_context::create_kernel_bpf_prog_in_userspace(int cmd,
							 union bpf_attr *attr,
							 size_t size)
{
	std::regex pattern(by_pass_kernel_verifier_pattern);
	int res = 0;
	if (!by_pass_kernel_verifier_pattern.empty() &&
	    std::regex_match(attr->prog_name, pattern)) {
		SPDLOG_INFO("By pass kernel verifier for program {}",
			    attr->prog_name);
		struct bpf_insn trival_prog_insns[] = {
			BPF_MOV64_IMM(BPF_REG_0, 0),
			BPF_EXIT_INSN(),
		};
		union bpf_attr new_attr = *attr;
		new_attr.insns = (uint64_t)(uintptr_t)trival_prog_insns;
		new_attr.insn_cnt = 2;
		new_attr.func_info_rec_size = 0;
		new_attr.func_info_cnt = 0;
		new_attr.func_info = 0;
		new_attr.line_info_rec_size = 0;
		new_attr.line_info_cnt = 0;
		new_attr.line_info = 0;
		res = orig_syscall_fn(__NR_bpf, (long)cmd,
				      (long)(uintptr_t)&new_attr, (long)size);
	} else {
		res = orig_syscall_fn(__NR_bpf, (long)cmd,
				      (long)(uintptr_t)attr, (long)size);
	}
	if (res < 0) {
		SPDLOG_ERROR("Failed to load program `{}`", attr->prog_name);
		return res;
	}
	int id = res;
	std::vector<ebpf_inst> insns;
	insns.resize(attr->insn_cnt);
	insns.assign((ebpf_inst *)(uintptr_t)attr->insns,
		     (ebpf_inst *)(uintptr_t)attr->insns + attr->insn_cnt);
	for (size_t i = 0; i < attr->insn_cnt; i++) {
		const struct ebpf_inst inst = insns[i];
		bool store = false;

		switch (inst.code) {
		case EBPF_OP_LDDW:
			if (inst.src_reg == 1 || inst.src_reg == 2) {
				bpf_map_info info = {};
				uint32_t info_len = sizeof(info);
				int res = bpf_obj_get_info_by_fd(
					inst.imm, &info, &info_len);
				if (res < 0) {
					SPDLOG_ERROR(
						"Failed to get map info for id {}",
						info.id);
					return -1;
				}
				SPDLOG_DEBUG(
					"relocate bpf prog insns for id {} in {}, "
					"lddw imm {} to map id {}",
					id, i, inst.imm, info.id);
				insns[i].imm = info.id;
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
	id = bpftime_progs_create(id /* let the shm alloc fd for us */,
				  insns.data(), (size_t)attr->insn_cnt,
				  attr->prog_name, attr->prog_type);
	SPDLOG_DEBUG("Loaded program `{}` id={}", attr->prog_name, id);
	return id;
}

long syscall_context::handle_sysbpf(int cmd, union bpf_attr *attr, size_t size)
{
	if (!enable_mock)
		return orig_syscall_fn(__NR_bpf, (long)cmd,
				       (long)(uintptr_t)attr, (long)size);
	try_startup();
	errno = 0;
	char *errmsg;
	switch (cmd) {
	case BPF_MAP_CREATE: {
		if (run_with_kernel) {
			SPDLOG_DEBUG("Creating kernel map");
			int fd = orig_syscall_fn(__NR_bpf, (long)cmd,
						 (long)(uintptr_t)attr,
						 (long)size);
			SPDLOG_DEBUG("Created kernel map {}", fd);
			return create_kernel_bpf_map(fd);
		}
		SPDLOG_DEBUG("Creating map");
		int id = bpftime_maps_create(
			-1 /* let the shm alloc fd for us */, attr->map_name,
			bpftime::bpf_map_attr{
				(int)attr->map_type,
				attr->key_size,
				attr->value_size,
				attr->max_entries,
				attr->map_flags,
				attr->map_ifindex,
				attr->btf_vmlinux_value_type_id,
				attr->btf_id,
				attr->btf_key_type_id,
				attr->btf_value_type_id,
				attr->map_extra,
			});
		SPDLOG_DEBUG(
			"Created map {}, type={}, name={}, key_size={}, value_size={}",
			id, attr->map_type, attr->map_name, attr->key_size,
			attr->value_size);
		return id;
	}
	case BPF_MAP_LOOKUP_ELEM: {
		SPDLOG_DEBUG("Looking up map {}", attr->map_fd);
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		// Note that bpftime_map_lookup_elem is adapted as a bpf helper,
		// meaning that it will *return* the address of the matched
		// value. But here the syscall has a different interface. Here
		// we should write the bytes of the matched value to the pointer
		// that user gave us. So here needs a memcpy to achive such
		// thing.
		auto value_ptr = bpftime_map_lookup_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key);
		if (value_ptr == nullptr) {
			errno = ENOENT;
			return -1;
		}
		memcpy((void *)(uintptr_t)attr->value, value_ptr,
		       bpftime_map_value_size_from_syscall(attr->map_fd));
		return 0;
	}
	case BPF_MAP_UPDATE_ELEM: {
		SPDLOG_DEBUG("Updating map");
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		return bpftime_map_update_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key,
			(const void *)(uintptr_t)attr->value,
			(uint64_t)attr->flags);
	}
	case BPF_MAP_DELETE_ELEM: {
		SPDLOG_DEBUG("Deleting map");
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		return bpftime_map_delete_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key);
	}
	case BPF_MAP_GET_NEXT_KEY: {
		SPDLOG_DEBUG("Getting next key");
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		return (long)(uintptr_t)bpftime_map_get_next_key(
			attr->map_fd, (const void *)(uintptr_t)attr->key,
			(void *)(uintptr_t)attr->next_key);
	}
	case BPF_PROG_LOAD:
		// Load a program?
		{
			SPDLOG_DEBUG(
				"Loading program `{}` license `{}` prog_type `{}` attach_type {} map_type {}",
				attr->prog_name,
				(const char *)(uintptr_t)attr->license,
				attr->prog_type, attr->attach_type,
				attr->map_type);
			if (run_with_kernel) {
				return create_kernel_bpf_prog_in_userspace(
					cmd, attr, size);
			}
			// tracepoint -> BPF_PROG_TYPE_TRACEPOINT
			// uprobe/uretprobe -> BPF_PROG_TYPE_SOCKET_FILTER
			std::optional<std::string> simple_section_name;
			if (attr->prog_type == BPF_PROG_TYPE_TRACEPOINT) {
				simple_section_name = "tracepoint";
			} else if (attr->prog_type ==
				   BPF_PROG_TYPE_SOCKET_FILTER) {
				simple_section_name = "uprobe";
			}
#ifdef ENABLE_BPFTIME_VERIFIER
			// Only do verification for tracepoint/uprobe/uretprobe
			if (simple_section_name.has_value()) {
				SPDLOG_DEBUG("Verying program {}",
					     attr->prog_name);
				auto result = verifier::verify_ebpf_program(
					(uint64_t *)(uintptr_t)attr->insns,
					(size_t)attr->insn_cnt,
					simple_section_name.value());
				if (result.has_value()) {
					std::ostringstream message;
					message << *result;
					// Print the program by bytes
					for (size_t i = 0; i < attr->insn_cnt;
					     i++) {
						uint64_t inst =
							((uint64_t *)(uintptr_t)
								 attr->insns)[i];
						message << std::setw(3)
							<< std::setfill('0')
							<< i << ": ";
						for (int j = 0; j < 8; j++) {
							message << std::hex
								<< std::uppercase
								<< std::setw(2)
								<< std::setfill(
									   '0')
								<< (inst & 0xff)
								<< " ";
							inst >>= 8;
						}
						message << std::endl;
					}
					SPDLOG_ERROR(
						"Failed to verify program: {}",
						message.str());
					errno = EINVAL;
					return -1;
				}
			}
#endif
			int id = bpftime_progs_create(
				-1 /* let the shm alloc fd for us */,
				(ebpf_inst *)(uintptr_t)attr->insns,
				(size_t)attr->insn_cnt, attr->prog_name,
				attr->prog_type);
			SPDLOG_DEBUG("Loaded program `{}` id={}",
				     attr->prog_name, id);
			return id;
		}
	case BPF_LINK_CREATE: {
		auto prog_fd = attr->link_create.prog_fd;
		auto target_fd = attr->link_create.target_fd;
		auto attach_type = attr->link_create.attach_type;
		SPDLOG_DEBUG("Creating link {} -> {}, attach type {}", prog_fd,
			     target_fd, attach_type);
		if (run_with_kernel && !bpftime_is_perf_event_fd(target_fd)) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		int id = bpftime_link_create(
			-1 /* let the shm alloc fd for us */,
			(bpf_link_create_args *)&attr->link_create);
		SPDLOG_DEBUG("Created link {}", id);
		if (bpftime_is_prog_fd(prog_fd) &&
		    bpftime_is_perf_event_fd(target_fd) &&
		    attach_type == BPF_PERF_EVENT) {
			auto cookie = attr->link_create.perf_event.bpf_cookie;
			SPDLOG_DEBUG(
				"Attaching perf event {} to prog {}, with bpf cookie {:x}",
				target_fd, prog_fd, cookie);
			bpftime_attach_perf_to_bpf_with_cookie(target_fd,
							       prog_fd, cookie);
		}
		return id;
	}
	case BPF_MAP_FREEZE: {
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		SPDLOG_DEBUG(
			"Calling bpf map freeze, but we didn't implement this");
		return 0;
	}
	case BPF_OBJ_GET_INFO_BY_FD: {
		if (run_with_kernel) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		SPDLOG_DEBUG("Getting info by fd");
		if (bpftime_is_map_fd(attr->info.bpf_fd)) {
			bpftime::bpf_map_attr map_attr;
			const char *map_name;
			bpftime::bpf_map_type map_type;
			int res = bpftime_map_get_info(attr->info.bpf_fd,
						       &map_attr, &map_name,
						       &map_type);
			if (res < 0) {
				errno = res;
				return -1;
			}
			auto ptr = (bpf_map_info *)((uintptr_t)attr->info.info);
			ptr->btf_id = map_attr.btf_id;
			ptr->btf_key_type_id = map_attr.btf_key_type_id;
			ptr->btf_value_type_id = map_attr.btf_value_type_id;
			ptr->type = (int)map_type;
			ptr->value_size = map_attr.value_size;
			ptr->btf_vmlinux_value_type_id =
				map_attr.btf_vmlinux_value_type_id;
			ptr->key_size = map_attr.key_size;
			ptr->id = attr->info.bpf_fd;
			ptr->ifindex = map_attr.ifindex;
			ptr->map_extra = map_attr.map_extra;
			// TODO: handle the rest info
			ptr->max_entries = map_attr.max_ents;
			ptr->map_flags = map_attr.flags;
			strncpy(ptr->name, map_name, sizeof(ptr->name) - 1);
		} else if (bpftime_is_prog_fd(attr->info.bpf_fd)) {
			auto ptr =
				(bpf_prog_info *)((uintptr_t)attr->info.info);
			ptr->id = attr->info.bpf_fd;
			// TODO: handle the rest info
			return 0;
		}
		return 0;
	}
	case BPF_PROG_ATTACH: {
		auto prog_fd = attr->attach_bpf_fd;
		auto target_fd = attr->target_fd;
		SPDLOG_DEBUG("BPF_PROG_ATTACH {} -> {}", prog_fd, target_fd);
		if (run_with_kernel && !bpftime_is_perf_event_fd(target_fd)) {
			return orig_syscall_fn(__NR_bpf, (long)cmd,
					       (long)(uintptr_t)attr,
					       (long)size);
		}
		int id = bpftime_attach_perf_to_bpf(target_fd, prog_fd);
		SPDLOG_DEBUG("Created attach {}", id);
		return id;
	}
	default:
		return orig_syscall_fn(__NR_bpf, (long)cmd,
				       (long)(uintptr_t)attr, (long)size);
	};
	return 0;
}
int syscall_context::handle_perfevent(perf_event_attr *attr, pid_t pid, int cpu,
				      int group_fd, unsigned long flags)
{
	if (!enable_mock)
		return orig_syscall_fn(__NR_perf_event_open,
				       (uint64_t)(uintptr_t)attr, (uint64_t)pid,
				       (uint64_t)cpu, (uint64_t)group_fd,
				       (uint64_t)flags);
	try_startup();
	if ((int)attr->type == determine_uprobe_perf_type()) {
		// NO legacy bpf types
		bool retprobe =
			attr->config & (1 << determine_uprobe_retprobe_bit());
		size_t ref_ctr_off =
			attr->config >> PERF_UPROBE_REF_CTR_OFFSET_SHIFT;
		const char *name = (const char *)(uintptr_t)attr->config1;
		uint64_t offset = attr->config2;
		SPDLOG_DEBUG(
			"Creating uprobe name {} offset {} retprobe {} ref_ctr_off {} attr->config={:x}",
			name, offset, retprobe, ref_ctr_off, attr->config);
		int id = bpftime_uprobe_create(
			-1 /* let the shm alloc fd for us */, pid, name, offset,
			retprobe, ref_ctr_off);
		// std::cout << "Created uprobe " << id << std::endl;
		SPDLOG_DEBUG("Created uprobe {}", id);
		return id;
	} else if ((int)attr->type ==
		   (int)bpf_event_type::PERF_TYPE_TRACEPOINT) {
		SPDLOG_DEBUG("Detected tracepoint perf event creation");
		int fd = bpftime_tracepoint_create(
			-1 /* let the shm alloc fd for us */, pid,
			(int32_t)attr->config);
		SPDLOG_DEBUG("Created tracepoint perf event with fd {}", fd);
		return fd;
	} else if ((int)attr->type == (int)bpf_event_type::PERF_TYPE_SOFTWARE) {
		SPDLOG_DEBUG("Detected software perf event creation");
		int fd = bpftime_add_software_perf_event(cpu, attr->sample_type,
							 attr->config);
		SPDLOG_DEBUG("Created software perf event with fd {}", fd);
		return fd;
	} else if ((int)attr->type == (int)bpf_event_type::BPF_TYPE_UREPLACE) {
		SPDLOG_DEBUG("Detected BPF_TYPE_UREPLACE");
		const char *name = (const char *)(uintptr_t)attr->config1;
		uint64_t offset = attr->config2;
		int fd = bpftime_add_ureplace_or_override(
			-1 /* let the shm alloc fd for us */, pid, name, offset,
			true);
		SPDLOG_DEBUG("Created ureplace with fd {}", fd);
		return fd;
	} else if ((int)attr->type ==
		   (int)bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE) {
		SPDLOG_DEBUG("Detected BPF_TYPE_UPROBE_OVERRIDE");
		const char *name = (const char *)(uintptr_t)attr->config1;
		uint64_t offset = attr->config2;
		int fd = bpftime_add_ureplace_or_override(
			-1 /* let the shm alloc fd for us */, pid, name, offset,
			false);
		SPDLOG_DEBUG("Created ufilter with fd {}", fd);
		return fd;
	}
	SPDLOG_INFO("Calling original perf event open");
	return orig_syscall_fn(__NR_perf_event_open, (uint64_t)(uintptr_t)attr,
			       (uint64_t)pid, (uint64_t)cpu, (uint64_t)group_fd,
			       (uint64_t)flags);
}

void *syscall_context::handle_mmap(void *addr, size_t length, int prot,
				   int flags, int fd, off64_t offset)
{
	if (!enable_mock || run_with_kernel)
		return orig_mmap_fn(addr, length, prot, flags, fd, offset);
	try_startup();
	SPDLOG_DEBUG("Called normal mmap");
	return handle_mmap64(addr, length, prot, flags, fd, offset);
}

void *syscall_context::handle_mmap64(void *addr, size_t length, int prot,
				     int flags, int fd, off64_t offset)
{
	if (!enable_mock || run_with_kernel)
		return orig_mmap64_fn(addr, length, prot, flags, fd, offset);
	try_startup();
	SPDLOG_DEBUG("Calling mocked mmap64");
	if (fd != -1 && bpftime_is_ringbuf_map(fd)) {
		SPDLOG_DEBUG("Entering mmap64 handling for ringbuf fd: {}", fd);
		if (prot == (PROT_WRITE | PROT_READ)) {
			if (auto ptr = bpftime_get_ringbuf_consumer_page(fd);
			    ptr != nullptr) {
				SPDLOG_DEBUG(
					"Mapping consumer page {} to ringbuf fd {}",
					ptr, fd);
				mocked_mmap_values.insert((uintptr_t)ptr);
				return ptr;
			}
		} else if (prot == (PROT_READ)) {
			if (auto ptr = bpftime_get_ringbuf_producer_page(fd);
			    ptr != nullptr) {
				SPDLOG_DEBUG(
					"Mapping producer page {} to ringbuf fd {}",
					ptr, fd);

				mocked_mmap_values.insert((uintptr_t)ptr);
				return ptr;
			}
		}
	} else if (fd != -1 && bpftime_is_array_map(fd)) {
		SPDLOG_DEBUG("Entering mmap64 which handled array map");
		if (auto val = bpftime_get_array_map_raw_data(fd);
		    val != nullptr) {
			mocked_mmap_values.insert((uintptr_t)val);
			return val;
		}
	} else if (fd != -1 && bpftime_is_software_perf_event(fd)) {
		SPDLOG_DEBUG(
			"Entering mocked mmap64: software perf event handler");
		if (auto ptr = bpftime_get_software_perf_event_raw_buffer(
			    fd, length);
		    ptr != nullptr) {
			mocked_mmap_values.insert((uintptr_t)ptr);
			return ptr;
		}
	}
	SPDLOG_DEBUG(
		"Calling original mmap64: addr={}, length={}, prot={}, flags={}, fd={}, offset={}",
		addr, length, prot, flags, fd, offset);
	auto ptr = orig_mmap64_fn(addr, length, prot | PROT_WRITE,
				  flags | MAP_ANONYMOUS, -1, 0);
	return orig_mmap64_fn(addr, length, prot, flags, fd, offset);
}

int syscall_context::handle_ioctl(int fd, unsigned long req, int data)
{
	if (!enable_mock)
		return orig_ioctl_fn(fd, req, data);
	try_startup();
	int res;
	if (req == PERF_EVENT_IOC_ENABLE) {
		SPDLOG_DEBUG("Enabling perf event {}", fd);
		if (run_with_kernel && !bpftime_is_perf_event_fd(fd)) {
			return orig_ioctl_fn(fd, req, data);
		}
		res = bpftime_perf_event_enable(fd);
		if (res >= 0)
			return res;
		spdlog::warn(
			"Failed to call mocked ioctl PERF_EVENT_IOC_ENABLE: {}",
			res);
	} else if (req == PERF_EVENT_IOC_DISABLE) {
		SPDLOG_DEBUG("Disabling perf event {}", fd);
		if (run_with_kernel && !bpftime_is_perf_event_fd(fd)) {
			return orig_ioctl_fn(fd, req, data);
		}
		res = bpftime_perf_event_disable(fd);
		if (res >= 0)
			return res;
		spdlog::warn(
			"Failed to call mocked ioctl PERF_EVENT_IOC_DISABLE: {}",
			res);
	} else if (req == PERF_EVENT_IOC_SET_BPF) {
		SPDLOG_DEBUG("Setting bpf for perf event {} and bpf {}", fd,
			     data);
		if (run_with_kernel && !bpftime_is_perf_event_fd(fd)) {
			return orig_ioctl_fn(fd, req, data);
		}
		res = bpftime_attach_perf_to_bpf(fd, data);
		if (res >= 0)
			return res;
		spdlog::warn(
			"Failed to call mocked ioctl PERF_EVENT_IOC_SET_BPF: {}",
			res);
	}
	spdlog::warn("Calling original ioctl: {} {} {}", fd, req, data);
	return orig_ioctl_fn(fd, req, data);
}

int syscall_context::handle_epoll_create1(int flags)
{
	if (!enable_mock || run_with_kernel)
		return orig_epoll_create1_fn(flags);
	try_startup();
	return bpftime_epoll_create();
}

int syscall_context::handle_epoll_ctl(int epfd, int op, int fd,
				      epoll_event *evt)
{
	if (!enable_mock || run_with_kernel)
		return orig_epoll_ctl_fn(epfd, op, fd, evt);
	try_startup();
	if (op == EPOLL_CTL_ADD) {
		if (bpftime_is_ringbuf_map(fd)) {
			int err = bpftime_add_ringbuf_fd_to_epoll(fd, epfd,
								  evt->data);
			if (err == 0) {
				return err;
			}
		} else if (bpftime_is_software_perf_event(fd)) {
			int err = bpftime_add_software_perf_event_fd_to_epoll(
				fd, epfd, evt->data);
			if (err == 0)
				return err;

		} else {
			spdlog::warn(
				"Unsupported map fd for mocked epoll_ctl: {}, call the original one..",
				fd);
		}
	}

	return orig_epoll_ctl_fn(epfd, op, fd, evt);
}

int syscall_context::handle_epoll_wait(int epfd, epoll_event *evt,
				       int maxevents, int timeout)
{
	if (!enable_mock || run_with_kernel)
		orig_epoll_wait_fn(epfd, evt, maxevents, timeout);
	try_startup();
	if (bpftime_is_epoll_handler(epfd)) {
		int ret = bpftime_epoll_wait(epfd, evt, maxevents, timeout);
		return ret;
	}
	return orig_epoll_wait_fn(epfd, evt, maxevents, timeout);
}

int syscall_context::handle_munmap(void *addr, size_t size)
{
	if (!enable_mock || run_with_kernel)
		orig_munmap_fn(addr, size);
	try_startup();
	if (auto itr = mocked_mmap_values.find((uintptr_t)addr);
	    itr != mocked_mmap_values.end()) {
		SPDLOG_DEBUG("Handling munmap of mocked addr: {:x}, size {}",
			     (uintptr_t)addr, size);
		mocked_mmap_values.erase(itr);
		return 0;
	} else {
		return orig_munmap_fn(addr, size);
	}
}

FILE *syscall_context::handle_fopen(const char *pathname, const char *flags)
{
	if (!enable_mock)
		return orig_fopen_fn(pathname, flags);
	try_startup();
	if (auto mocker = create_mocked_file_based_on_full_path(pathname);
	    mocker) {
		bpftime_lock_guard _guard(this->mocked_file_lock);
		char filename_buf[] = "/tmp/bpftime-mock.XXXXXX";
		int fake_fd = mkstemp(filename_buf);
		if (fake_fd < 0) {
			SPDLOG_WARN("Unable to create mock fd: {}", errno);
			return orig_fopen_fn(pathname, flags);
		}
		auto itr =
			this->mocked_files.emplace(fake_fd, std::move(*mocker))
				.first;
		FILE *replacement_fp = fopen(filename_buf, "r");

		itr->second->replacement_file = replacement_fp;
		auto size_written = write(fake_fd, itr->second->buf.c_str(),
					  itr->second->buf.size());
		SPDLOG_DEBUG(
			"Created fake fd {}, replacement fp {:x}, written {} bytes",
			fake_fd, (uintptr_t)replacement_fp, size_written);
		return replacement_fp;
	}
	return orig_fopen_fn(pathname, flags);
}
