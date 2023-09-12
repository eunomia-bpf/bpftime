#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <chrono>
#include <cstdarg>
#ifndef __x86_64__
#error "This poc only works on x86_64"
#endif

#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <dlfcn.h>
#include <linux/perf_event.h>
#include "wrapper.hpp"
#include <sys/mman.h>
#include <iterator>
#include <variant>
#include <asm-generic/errno-base.h>
#include <sys/epoll.h>

using syscall_fn = long (*)(long, ...);
using close_fn = int (*)(int);
using mmap64_fn = void *(*)(void *, size_t, int, int, int, off64_t);
using ioctl_fn = int (*)(int fd, unsigned long req, void *);
using epoll_craete1_fn = int (*)(int);
using epoll_ctl_fn = int (*)(int, int, int, struct epoll_event *);
using epoll_wait_fn = int (*)(int, struct epoll_event *, int, int);

static syscall_fn orig_syscall_fn = nullptr;
static close_fn orig_close_fn = nullptr;
static mmap64_fn orig_mmap64_fn = nullptr;
static ioctl_fn orig_ioctl_fn = nullptr;
static epoll_craete1_fn orig_epoll_create1_fn = nullptr;
static epoll_ctl_fn orig_epoll_ctl_fn = nullptr;
static epoll_wait_fn orig_epoll_wait_fn = nullptr;

static int handle_close(int);
static long handle_sysbpf(int cmd, union bpf_attr *attr, size_t size);
static int handle_perfevent(perf_event_attr *attr, pid_t pid, int cpu,
			    int group_fd, unsigned long flags);
static void *handle_mmap64(void *addr, size_t length, int prot, int flags,
			   int fd, off_t offset);
static int handle_ioctl(int fd, unsigned long req, void *);
static int handle_epoll_create1(int);
static int handle_epoll_ctl(int epfd, int op, int fd, epoll_event *evt);
static int handle_epoll_wait(int epfd, epoll_event *evt, int maxevents,
			     int timeout);

extern "C" int epoll_wait(int epfd, epoll_event *evt, int maxevents,
			  int timeout)
{
	if (orig_epoll_wait_fn == nullptr) {
		orig_epoll_wait_fn =
			(epoll_wait_fn)dlsym(RTLD_NEXT, "epoll_wait");
	}
	std::cerr << "epoll_wait " << epfd << std::endl;
	return handle_epoll_wait(epfd, evt, maxevents, timeout);
}

extern "C" int epoll_ctl(int epfd, int op, int fd, epoll_event *evt)
{
	if (orig_epoll_ctl_fn == nullptr) {
		orig_epoll_ctl_fn = (epoll_ctl_fn)dlsym(RTLD_NEXT, "epoll_ctl");
	}
	std::cerr << "epoll_ctl " << epfd << " " << op << " " << fd << " "
		  << evt << std::endl;
	return handle_epoll_ctl(epfd, op, fd, evt);
}

extern "C" int epoll_create1(int flags)
{
	if (orig_epoll_create1_fn == nullptr) {
		orig_epoll_create1_fn =
			(epoll_craete1_fn)dlsym(RTLD_NEXT, "epoll_create1");
	}
	std::cerr << "epoll_create1 " << flags << std::endl;
	return handle_epoll_create1(flags);
}

extern "C" int ioctl(int fd, unsigned long req, void *data)
{
	if (orig_ioctl_fn == nullptr) {
		orig_ioctl_fn = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
	}
	std::cerr << "ioctl " << fd << " " << req << " " << data << std::endl;
	return handle_ioctl(fd, req, data);
}

extern "C" void *mmap64(void *addr, size_t length, int prot, int flags, int fd,
			off64_t offset)
{
	if (orig_mmap64_fn == nullptr) {
		orig_mmap64_fn = (mmap64_fn)dlsym(RTLD_NEXT, "mmap");
	}
	std::cerr << "Mmap64 " << addr << std::endl;
	return handle_mmap64(addr, length, prot, flags, fd, offset);
}

extern "C" int close(int fd)
{
	if (orig_close_fn == nullptr) {
		orig_close_fn = (close_fn)dlsym(RTLD_NEXT, "close");
	}
	std::cout << "Closing " << fd << std::endl;
	return handle_close(fd);
}

extern "C" long syscall(long sysno, ...)
{
	if (orig_syscall_fn == nullptr) {
		orig_syscall_fn = (syscall_fn)dlsym(RTLD_NEXT, "syscall");
	}
	// glibc directly reads the arguments without considering
	// the underlying argument number. So did us
	va_list args;
	va_start(args, sysno);
	long arg1 = va_arg(args, long);
	long arg2 = va_arg(args, long);
	long arg3 = va_arg(args, long);
	long arg4 = va_arg(args, long);
	long arg5 = va_arg(args, long);
	long arg6 = va_arg(args, long);
	va_end(args);
	if (sysno == __NR_bpf) {
		std::cout << "SYS_BPF"
			  << " " << arg1 << " " << arg2 << " " << arg3 << " "
			  << arg4 << " " << arg5 << " " << arg6 << std::endl;
		int cmd = (int)arg1;
		auto attr = (union bpf_attr *)(uintptr_t)arg2;
		auto size = (size_t)arg3;
		return handle_sysbpf(cmd, attr, size);
	} else if (sysno == __NR_perf_event_open) {
		std::cout << "SYS_PERF_EVENT_OPEN"
			  << " " << arg1 << " " << arg2 << " " << arg3 << " "
			  << arg4 << " " << arg5 << " " << arg6 << std::endl;
		return handle_perfevent((perf_event_attr *)(uintptr_t)arg1,
					(pid_t)arg2, (int)arg3, (int)arg4,
					(unsigned long)arg5);
	} else if (sysno == __NR_ioctl) {
		std::cout << "SYS_IOCTL"
			  << " " << arg1 << " " << arg2 << " " << arg3 << " "
			  << arg4 << " " << arg5 << " " << arg6 << std::endl;
	}
	return orig_syscall_fn(sysno, arg1, arg2, arg3, arg4, arg5, arg6);
}

static int handle_close(int fd)
{
	if (auto itr = objs.find(fd); itr != objs.end()) {
		std::cerr << "Destroy ebpf obj: " << fd << " type "
			  << name_mapping[itr->second->index()] << std::endl;
		objs.erase(itr);
		return 0;
	} else {
		return orig_close_fn(fd);
	}
}

static long handle_sysbpf(int cmd, union bpf_attr *attr, size_t size)
{
	errno = 0;
	char *errmsg;
	switch (cmd) {
	case BPF_PROG_LOAD:
		// Load a program?
		{
			std::cerr << "Loading program `" << attr->prog_name
				  << "` license `"
				  << (const char *)(uintptr_t)attr->license
				  << "`" << std::endl;
			EbpfProgWrapper prog;

			if (int err = ebpf_load(prog.vm.get(),
						(void *)(uintptr_t)attr->insns,
						(uint32_t)attr->insn_cnt * 8,
						&errmsg);
			    err < 0) {
				std::cerr << "Failed to load code into vm, err="
					  << err << " message="
					  << (errmsg == nullptr ? "<unknown>" :
								  errmsg)
					  << std::endl;
				if (errmsg)
					free(errmsg);
				errno = EINVAL;
				return -1;
			}
			auto id = next_fd.fetch_add(1);
			objs.emplace(
				id, std::make_unique<EbpfObj>(std::move(prog)));
			std::cerr << "Loaded program `" << attr->prog_name
				  << "` id =" << id << std::endl;
			return id;
		}
	case BPF_LINK_CREATE: {
		auto prog_fd = attr->link_create.prog_fd;
		auto target_fd = attr->link_create.target_fd;
		if (auto itr = objs.find(prog_fd); itr != objs.end()) {
			if (!std::holds_alternative<EbpfProgWrapper>(
				    *itr->second)) {
				std::cerr << "prog fd " << prog_fd
					  << " is not program" << std::endl;
				errno = EBADF;
				return -1;
			}
		} else {
			std::cerr << "No prog fd " << prog_fd << " found"
				  << std::endl;
			errno = EBADF;
			return -1;
		}
		if (auto itr = objs.find(target_fd); itr != objs.end()) {
			if (!std::holds_alternative<PerfEventWrapper>(
				    *itr->second)) {
				std::cerr << "target fd " << target_fd
					  << " is not perf event" << std::endl;
				errno = EBADF;
				return -1;
			}
		} else {
			std::cerr << "No perf event fd " << target_fd
				  << " found" << std::endl;
			// Return BADF to simulate behavior of kernel
			errno = EBADF;
			return -1;
		}
		auto id = next_fd.fetch_add(1);
		objs.emplace(id, std::make_unique<EbpfObj>(
					 BpfLinkWrapper(prog_fd, target_fd)));
		return id;
	}
	case BPF_MAP_CREATE: {
		EbpfMapWrapper map((enum bpf_map_type)attr->map_type,
				   (uint32_t)attr->key_size,
				   (uint32_t)attr->value_size,
				   (uint32_t)attr->max_entries,
				   (uint64_t)attr->flags, attr->map_name);
		map.btf_id = attr->btf_id;
		map.btf_key_type_id = attr->btf_key_type_id;
		map.btf_value_type_id = attr->btf_value_type_id;
		map.btf_vmlinux_value_type_id = attr->btf_vmlinux_value_type_id;
		map.ifindex = attr->map_ifindex;
		map.map_extra = attr->map_extra;
		auto id = next_fd.fetch_add(1);
		objs.emplace(id, std::make_unique<EbpfObj>(std::move(map)));
		return id;
	}
	case BPF_MAP_UPDATE_ELEM: {
		if (auto itr = objs.find(attr->map_fd); itr != objs.end()) {
			if (std::holds_alternative<EbpfMapWrapper>(
				    *itr->second)) {
				auto &map =
					std::get<EbpfMapWrapper>(*itr->second);
				int err = map.mapUpdate(
					(const void *)(uintptr_t)(attr->key),
					(const void *)(uintptr_t)attr->value,
					(uint64_t)attr->flags);
				if (err < 0) {
					errno = -err;
					return -1;
				}
				return err;
			} else {
				errno = EINVAL;
				return -1;
			}
		} else {
			errno = ENOENT;
			return -1;
		}
	}
	case BPF_MAP_FREEZE: {
		if (auto itr = objs.find(attr->map_fd); itr != objs.end()) {
			if (std::holds_alternative<EbpfMapWrapper>(
				    *itr->second)) {
				auto &map =
					std::get<EbpfMapWrapper>(*itr->second);
				map.frozen = true;
				return 0;
			} else {
				errno = EINVAL;
				return -1;
			}
		} else {
			errno = ENOENT;
			return -1;
		}
	}
	case BPF_OBJ_GET_INFO_BY_FD: {
		if (auto itr = objs.find(attr->info.bpf_fd);
		    itr != objs.end()) {
			if (std::holds_alternative<EbpfMapWrapper>(
				    *itr->second)) {
				auto &map =
					std::get<EbpfMapWrapper>(*itr->second);
				auto ptr =
					(bpf_map_info *)((uintptr_t)attr->info
								 .info);
				ptr->btf_id = map.btf_id;
				ptr->btf_key_type_id = map.btf_key_type_id;
				ptr->btf_value_type_id = map.btf_value_type_id;
				ptr->type = map.type;
				ptr->value_size = map.value_size;
				ptr->btf_vmlinux_value_type_id =
					map.btf_vmlinux_value_type_id;
				ptr->key_size = map.key_size;
				ptr->id = attr->info.bpf_fd;
				ptr->ifindex = map.ifindex;
				ptr->map_extra = map.map_extra;
				ptr->max_entries = map.max_entries;
				ptr->netns_dev = map.netns_dev;
				ptr->netns_ino = map.netns_ino;
				ptr->map_flags = map.flags;
				strncpy(ptr->name, map.name.c_str(),
					sizeof(ptr->name) - 1);
				return 0;
			} else {
				std::cerr
					<< "Currently only supports get info of maps"
					<< std::endl;
				errno = ENOTSUP;
				return -1;
			}
		} else {
			errno = ENOENT;
			return -1;
		}
		break;
	}
	default:
		return orig_syscall_fn(__NR_bpf, (long)cmd,
				       (long)(uintptr_t)attr, (long)size);
	};
	return 0;
}

static int handle_perfevent(perf_event_attr *attr, pid_t pid, int cpu,
			    int group_fd, unsigned long flags)
{
	if (attr->type == PERF_TYPE_TRACEPOINT) {
		auto id = next_fd.fetch_add(1);
		objs.emplace(id, std::make_unique<EbpfObj>(PerfEventWrapper()));
		return id;
	} else {
		return orig_syscall_fn(__NR_perf_event_open,
				       (uint64_t)(uintptr_t)attr, (uint64_t)pid,
				       (uint64_t)cpu, (uint64_t)group_fd,
				       (uint64_t)flags);
	}
	return 0;
}

static void *handle_mmap64(void *addr, size_t length, int prot, int flags,
			   int fd, off64_t offset)
{
	if (auto itr = objs.find(fd); itr != objs.end()) {
		if (std::holds_alternative<EbpfMapWrapper>(*itr->second)) {
			auto ptr =
				orig_mmap64_fn(addr, length, prot | PROT_WRITE,
					       flags | MAP_ANONYMOUS, -1, 0);
			auto &wrapper = std::get<EbpfMapWrapper>(*itr->second);
			if (std::holds_alternative<ArrayMapImpl>(
				    wrapper.impl)) {
				auto &map =
					std::get<ArrayMapImpl>(wrapper.impl);
				for (size_t i = 0; i < map.size(); i++) {
					std::copy(
						map[i].begin(), map[i].end(),
						(uint8_t *)ptr +
							i * wrapper.value_size);
				}
			} else if (std::holds_alternative<RingBufMapImpl>(
					   wrapper.impl)) {
				auto impl =
					std::get<RingBufMapImpl>(wrapper.impl);
				if (prot == (PROT_WRITE | PROT_READ)) {
					impl->consumer_pos =
						(unsigned long *)ptr;
					std::cerr << "Ringbuf " << fd
						  << " writeable ptr " << ptr
						  << std::endl;
					memset(impl->consumer_pos, 0, length);
				} else if (prot == (PROT_READ)) {
					impl->producer_pos =
						(unsigned long *)ptr;
					impl->data =
						(uint8_t *)((uintptr_t)impl
								    ->producer_pos +
							    getpagesize());
					std::cerr << "Ringbuf " << fd
						  << " readonly ptr " << ptr
						  << std::endl;
					memset(impl->producer_pos, 0, length);
				}
			} else {
				std::cerr
					<< "Currently only supports mapping array backed fds"
					<< std::endl;
			}
			return (void *)ptr;
		}
	}
	return orig_mmap64_fn(addr, length, prot, flags, fd, offset);
}

static int handle_ioctl(int fd, unsigned long req, void *data)
{
	if (auto itr = objs.find(fd); itr != objs.end()) {
		if (std::holds_alternative<PerfEventWrapper>(*itr->second)) {
			auto &perf = std::get<PerfEventWrapper>(*itr->second);
			perf.enabled = true;
			return 0;
		} else {
			errno = EINVAL;
			return -1;
		}
	}
	return orig_ioctl_fn(fd, req, data);
}

static int handle_epoll_create1(int flags)
{
	int fd = next_fd.fetch_add(1);
	objs.emplace(fd, std::make_unique<EbpfObj>(EpollWrapper()));
	return fd;
}

static int handle_epoll_ctl(int epfd, int op, int fd, epoll_event *evt)
{
	if (auto itr = objs.find(epfd); itr != objs.end()) {
		std::optional<std::weak_ptr<RingBuffer> > rb_ptr;
		if (auto itr_fd = objs.find(fd); itr_fd != objs.end()) {
			if (std::holds_alternative<EbpfMapWrapper>(
				    *itr_fd->second)) {
				auto &map = std::get<EbpfMapWrapper>(
					*itr_fd->second);
				if (map.type != BPF_MAP_TYPE_RINGBUF) {
					std::cerr << "fd " << fd
						  << " is not a ringbuf map"
						  << std::endl;
					errno = EINVAL;

					return -1;
				}
				rb_ptr = std::get<RingBufMapImpl>(map.impl);
			} else {
				std::cerr << "fd " << fd << " is not a map"
					  << std::endl;
				errno = EINVAL;
				return -1;
			}
		} else {
			std::cerr << "Bad fd: " << fd << std::endl;
			errno = EINVAL;
			return -1;
		}
		if (std::holds_alternative<EpollWrapper>(*itr->second)) {
			auto &ep = std::get<EpollWrapper>(*itr->second);
			if (op == EPOLL_CTL_ADD) {
				ep.rbs.push_back(rb_ptr.value());
				return 0;
			} else {
				std::cerr << "Bad epoll op " << op << std::endl;
				errno = EINVAL;
				return -1;
			}
		} else {
			errno = EINVAL;
			return -1;
		}
	}
	return orig_epoll_ctl_fn(epfd, op, fd, evt);
}

static int handle_epoll_wait(int epfd, epoll_event *evt, int maxevents,
			     int timeout)
{
	if (auto itr = objs.find(epfd); itr != objs.end()) {
		if (std::holds_alternative<EpollWrapper>(*itr->second)) {
			auto &ep = std::get<EpollWrapper>(*itr->second);
			using namespace std::chrono;

			auto start = high_resolution_clock::now();
			int next_id = 0;
			while (next_id < maxevents) {
				auto now = high_resolution_clock::now();
				auto elasped = duration_cast<milliseconds>(
					now - start);
				if (elasped.count() >= 100)
					break;
				int idx = 0;
				for (auto p : ep.rbs) {
					if (auto ptr = p.lock(); ptr) {
						if (ptr->has_data()) {
							auto &next_event =
								evt[next_id++];
							next_event.events =
								EPOLLIN;
							next_event.data.fd =
								idx;
						}
					}
					idx++;
				}
			}
			return next_id;
		} else {
			errno = EINVAL;
			return -1;
		}
	}
	return orig_epoll_wait_fn(epfd, evt, maxevents, timeout);
}
