#include "syscall_context.hpp"
#include "bpftime_shm.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <linux/bpf.h>
#include "syscall_server_utils.hpp"
#include <optional>
#include <sys/epoll.h>
using namespace bpftime;

int syscall_context::handle_close(int fd)
{
	bpftime_close(fd);
	return orig_close_fn(fd);
}

long syscall_context::handle_sysbpf(int cmd, union bpf_attr *attr, size_t size)
{
	start_up();
	errno = 0;
	char *errmsg;
	switch (cmd) {
	case BPF_MAP_CREATE: {
		spdlog::debug("Creating map");
		int id = bpftime_maps_create(
			attr->map_name, bpftime::bpf_map_attr{
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
		spdlog::debug(
			"Created map {}, type={}, name={}, key_size={}, value_size={}",
			id, attr->map_type, attr->map_name, attr->key_size,
			attr->value_size);
		return id;
	}
	case BPF_MAP_LOOKUP_ELEM: {
		spdlog::debug("Looking up map {}");
		// Note that bpftime_map_lookup_elem is adapted as a bpf helper,
		// meaning that it will *return* the address of the matched
		// value. But here the syscall has a different interface. Here
		// we should write the bytes of the matched value to the pointer
		// that user gave us. So here needs a memcpy to achive such
		// thing.
		auto value_ptr = bpftime_map_lookup_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key);
		memcpy((void *)(uintptr_t)attr->value, value_ptr,
		       bpftime_map_value_size(attr->map_fd));
		return 0;
	}
	case BPF_MAP_UPDATE_ELEM: {
		spdlog::debug("Updating map");
		return bpftime_map_update_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key,
			(const void *)(uintptr_t)attr->value,
			(uint64_t)attr->flags);
	}
	case BPF_MAP_DELETE_ELEM: {
		spdlog::debug("Deleting map");
		return bpftime_map_delete_elem(
			attr->map_fd, (const void *)(uintptr_t)attr->key);
	}
	case BPF_MAP_GET_NEXT_KEY: {
		spdlog::debug("Getting next key");
		return (long)(uintptr_t)bpftime_map_get_next_key(
			attr->map_fd, (const void *)(uintptr_t)attr->key,
			(void *)(uintptr_t)attr->next_key);
	}
	case BPF_PROG_LOAD:
		// Load a program?
		{
			spdlog::debug(
				"Loading program `{}` license `{}` prog_type `{}` attach_type {} map_type {}",
				attr->prog_name,
				(const char *)(uintptr_t)attr->license,
				attr->prog_type, attr->attach_type,
				attr->map_type);

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
				spdlog::debug("Verying program {}",
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
					spdlog::error(
						"Failed to verify program: {}",
						message.str());
					errno = EINVAL;
					return -1;
				}
			}
#endif
			int id = bpftime_progs_create(
				(ebpf_inst *)(uintptr_t)attr->insns,
				(size_t)attr->insn_cnt, attr->prog_name,
				attr->prog_type);
			spdlog::debug("Loaded program `{}` id={}",
				      attr->prog_name, id);
			return id;
		}
	case BPF_LINK_CREATE: {
		auto prog_fd = attr->link_create.prog_fd;
		auto target_fd = attr->link_create.target_fd;
		spdlog::debug("Creating link {} -> {}", prog_fd, target_fd);
		int id = bpftime_link_create(prog_fd, target_fd);
		spdlog::debug("Created link {}", id);
		return id;
	}
	case BPF_MAP_FREEZE: {
		spdlog::debug(
			"Calling bpf map freeze, but we didn't implement this");
		return 0;
	}
	case BPF_OBJ_GET_INFO_BY_FD: {
		spdlog::debug("Getting info by fd");
		bpftime::bpf_map_attr map_attr;
		const char *map_name;
		int map_type;
		int res = bpftime_map_get_info(attr->info.bpf_fd, &map_attr,
					       &map_name, &map_type);
		if (res < 0) {
			errno = res;
			return -1;
		}
		auto ptr = (bpf_map_info *)((uintptr_t)attr->info.info);
		ptr->btf_id = map_attr.btf_id;
		ptr->btf_key_type_id = map_attr.btf_key_type_id;
		ptr->btf_value_type_id = map_attr.btf_value_type_id;
		ptr->type = map_type;
		ptr->value_size = map_attr.value_size;
		ptr->btf_vmlinux_value_type_id =
			map_attr.btf_vmlinux_value_type_id;
		ptr->key_size = map_attr.key_size;
		ptr->id = attr->info.bpf_fd;
		ptr->ifindex = map_attr.ifindex;
		ptr->map_extra = map_attr.map_extra;
		// 		ptr->netns_dev = map.netns_dev;
		// 		ptr->netns_ino = map.netns_ino;
		ptr->max_entries = map_attr.max_ents;
		ptr->map_flags = map_attr.flags;
		strncpy(ptr->name, map_name, sizeof(ptr->name) - 1);
		return 0;
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
	if ((int)attr->type == determine_uprobe_perf_type()) {
		// NO legacy bpf types
		bool retprobe = attr->config & determine_uprobe_retprobe_bit();
		size_t ref_ctr_off =
			attr->config >> PERF_UPROBE_REF_CTR_OFFSET_SHIFT;
		const char *name = (const char *)(uintptr_t)attr->config1;
		uint64_t offset = attr->config2;
		spdlog::debug(
			"Creating uprobe name {} offset {} retprove {} ref_ctr_off {}",
			name, offset, retprobe, ref_ctr_off);
		int id = bpftime_uprobe_create(pid, name, offset, retprobe,
					       ref_ctr_off);
		// std::cout << "Created uprobe " << id << std::endl;
		spdlog::debug("Created uprobe {}", id);
		return id;
	} else if ((int)attr->type ==
		   (int)bpf_perf_event_handler::bpf_event_type::
			   PERF_TYPE_TRACEPOINT) {
		spdlog::debug("Detected tracepoint perf event creation");
		int fd = bpftime_tracepoint_create(pid, (int32_t)attr->config);
		spdlog::debug("Created tracepoint perf event with fd {}", fd);
		return fd;
	} else if ((int)attr->type ==
		   (int)bpf_perf_event_handler::bpf_event_type::
			   PERF_TYPE_SOFTWARE) {
		spdlog::debug("Detected software perf event creation");
		int fd = bpftime_add_software_perf_event(cpu, attr->sample_type,
							 attr->config);
		spdlog::debug("Created software perf event with fd {}", fd);
		return fd;
	}
	spdlog::warn("Calling original perf event open");
	return orig_syscall_fn(__NR_perf_event_open, (uint64_t)(uintptr_t)attr,
			       (uint64_t)pid, (uint64_t)cpu, (uint64_t)group_fd,
			       (uint64_t)flags);
}

void *syscall_context::handle_mmap64(void *addr, size_t length, int prot,
				     int flags, int fd, off64_t offset)
{
	spdlog::debug("Calling mocked mmap64");
	if (fd != -1 && bpftime_is_ringbuf_map(fd)) {
		spdlog::debug("Entering mmap64 handling for ringbuf fd: {}",
			      fd);
		if (prot == (PROT_WRITE | PROT_READ)) {
			if (auto ptr = bpftime_get_ringbuf_consumer_page(fd);
			    ptr != nullptr) {
				spdlog::debug(
					"Mapping consumer page {} to ringbuf fd {}",
					ptr, fd);

				return ptr;
			}
		} else if (prot == (PROT_READ)) {
			if (auto ptr = bpftime_get_ringbuf_producer_page(fd);
			    ptr != nullptr) {
				spdlog::debug(
					"Mapping producer page {} to ringbuf fd {}",
					ptr, fd);

				return ptr;
			}
		}
	} else if (fd != -1 && bpftime_is_array_map(fd)) {
		spdlog::debug("Entering mmap64 which handled array map");
		if (auto val = bpftime_get_array_map_raw_data(fd);
		    val != nullptr) {
			return val;
		}
	} else if (fd != -1 && bpftime_is_software_perf_event(fd)) {
		spdlog::debug(
			"Entering mocked mmap64: software perf event handler");
		return bpftime_get_software_perf_event_raw_buffer(fd, length);
	}

	auto ptr = orig_mmap64_fn(addr, length, prot | PROT_WRITE,
				  flags | MAP_ANONYMOUS, -1, 0);
	spdlog::debug(
		"Calling original mmap64: addr={}, length={}, prot={}, flags={}, fd={}, offset={}",
		addr, length, prot, flags, fd, offset);
	return orig_mmap64_fn(addr, length, prot, flags, fd, offset);
}

int syscall_context::handle_ioctl(int fd, unsigned long req, int data)
{
	int res;
	switch (req) {
	case PERF_EVENT_IOC_ENABLE: {
		spdlog::debug("Enabling perf event {}", fd);
		res = bpftime_attach_enable(fd);
		if (res < 0) {
			return orig_ioctl_fn(fd, req, data);
		}
		return res;
	}
	case PERF_EVENT_IOC_SET_BPF: {
		spdlog::debug("Setting bpf for perf event {} and bpf {}", fd,
			      data);
		res = bpftime_attach_perf_to_bpf(fd, data);
		if (res < 0) {
			return orig_ioctl_fn(fd, req, data);
		}
		return res;
	}
	default:
		return orig_ioctl_fn(fd, req, data);
	}
	return 0;
}

int syscall_context::handle_epoll_create1(int flags)
{
	return bpftime_epoll_create();
}

int syscall_context::handle_epoll_ctl(int epfd, int op, int fd,
				      epoll_event *evt)
{
	if (op == EPOLL_CTL_ADD) {
		if (bpftime_is_ringbuf_map(fd)) {
			int err = bpftime_add_ringbuf_fd_to_epoll(fd, epfd);
			if (err == 0) {
				return err;
			}
		} else if (bpftime_is_software_perf_event(fd)) {
			int err = bpftime_add_software_perf_event_fd_to_epoll(
				fd, epfd);
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
	if (bpftime_is_epoll_handler(epfd)) {
		std::vector<int> fd_buf(maxevents);
		int ret = bpftime_ringbuf_poll(epfd, &fd_buf[0], maxevents,
					       timeout);
		if (ret >= 0) {
			for (size_t i = 0; i < (size_t)ret; i++) {
				evt[i] = epoll_event{
					.events = EPOLLIN,
					.data = { .fd = fd_buf[i] }
				};
			}
			return ret;
		}
	}
	return orig_epoll_wait_fn(epfd, evt, maxevents, timeout);
}
