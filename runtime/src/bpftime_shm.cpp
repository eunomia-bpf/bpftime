#include "handler/epoll_handler.hpp"
#include "handler/map_handler.hpp"
#include "spdlog/spdlog.h"
#include <asm-generic/errno-base.h>
#include <bpftime_shm_internal.hpp>
#include <variant>
namespace bpftime
{

bpftime_shm_holder shm_holder;

static __attribute__((destructor(1))) void __destroy_bpftime_shm_holder()
{
	shm_holder.global_shared_memory.~bpftime_shm();
}

// Check whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
bool bpftime_shm::check_syscall_trace_setup(int pid)
{
	return syscall_installed_pids->contains(pid);
}
// Set whether a certain pid was already equipped with syscall tracer
// Using a set stored in the shared memory
void bpftime_shm::set_syscall_trace_setup(int pid, bool whether)
{
	if (whether) {
		syscall_installed_pids->insert(pid);
	} else {
		syscall_installed_pids->erase(pid);
	}
}
uint32_t bpftime_shm::bpf_map_value_size(int fd) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.get_value_size();
}
const void *bpftime_shm::bpf_map_lookup_elem(int fd, const void *key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return nullptr;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_lookup_elem(key);
}

long bpftime_shm::bpf_update_elem(int fd, const void *key, const void *value,
				  uint64_t flags) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_update_elem(key, value, flags);
}

long bpftime_shm::bpf_delete_elem(int fd, const void *key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_delete_elem(key);
}

int bpftime_shm::bpf_map_get_next_key(int fd, const void *key,
				      void *next_key) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.bpf_map_get_next_key(key, next_key);
}

int bpftime_shm::add_uprobe(int pid, const char *name, uint64_t offset,
			    bool retprobe, size_t ref_ctr_off)
{
	int fd = open_fake_fd();
	manager->set_handler(
		fd,
		bpftime::bpf_perf_event_handler{ false, offset, pid, name,
						 ref_ctr_off, segment },
		segment);
	return fd;
}
int bpftime_shm::add_tracepoint(int pid, int32_t tracepoint_id)
{
	int fd = open_fake_fd();
	manager->set_handler(fd,
			     bpftime::bpf_perf_event_handler(pid, tracepoint_id,
							     segment),
			     segment);
	return fd;
}
int bpftime_shm::attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	if (!is_perf_fd(perf_fd) || !is_prog_fd(bpf_fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_prog_handler>(
		manager->get_handler(bpf_fd));
	handler.add_attach_fd(perf_fd);
	return 0;
}

int bpftime_shm::attach_enable(int fd) const
{
	if (!is_perf_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_perf_event_handler>(
		manager->get_handler(fd));
	handler.enable();
	return 0;
}

int bpftime_shm::add_ringbuf_to_epoll(int ringbuf_fd, int epoll_fd)
{
	if (!is_epoll_fd(epoll_fd)) {
		spdlog::error("Fd {} is expected to be an epoll fd", epoll_fd);
		errno = EINVAL;
		return -1;
	}
	auto &epoll_inst =
		std::get<epoll_handler>(manager->get_handler(epoll_fd));

	if (!is_map_fd(ringbuf_fd)) {
		spdlog::error("Fd {} is expected to be an map fd", ringbuf_fd);
		errno = EINVAL;
		return -1;
	}
	auto &map_inst =
		std::get<bpf_map_handler>(manager->get_handler(ringbuf_fd));

	auto ringbuf_map_impl = map_inst.try_get_ringbuf_map_impl();
	if (ringbuf_map_impl.has_value(); auto val = ringbuf_map_impl.value()) {
		epoll_inst.holding_ringbufs.push_back(
			val->create_impl_weak_ptr());
		spdlog::debug("Ringbuf {} added to epoll {}", ringbuf_fd,
			      epoll_fd);
		return 0;
	} else {
		errno = EINVAL;
		spdlog::error("Map fd {} is expected to be an ringbuf map",
			      ringbuf_fd);
		return -1;
	}
}
int bpftime_shm::epoll_create()
{
	int fd = open_fake_fd();
	if (manager->is_allocated(fd)) {
		spdlog::error(
			"Creating epoll instance, but fd {} is already occupied",
			fd);
		return -1;
	}
	manager->set_handler(fd, bpftime::epoll_handler(segment), segment);
	spdlog::debug("Epoll instance created: fd={}", fd);
	return fd;
}

bpftime::agent_config &bpftime_get_agent_config()
{
	return shm_holder.global_shared_memory.get_agent_config();
}

} // namespace bpftime

using namespace bpftime;

int bpftime_link_create(int prog_fd, int target_fd)
{
	return shm_holder.global_shared_memory.add_bpf_link(prog_fd, target_fd);
}

int bpftime_progs_create(const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
{
	return shm_holder.global_shared_memory.add_bpf_prog(
		insn, insn_cnt, prog_name, prog_type);
}

int bpftime_maps_create(const char *name, bpftime::bpf_map_attr attr)
{
	return shm_holder.global_shared_memory.add_bpf_map(name, attr);
}
uint32_t bpftime_map_value_size(int fd)
{
	return shm_holder.global_shared_memory.bpf_map_value_size(fd);
}

const void *bpftime_map_lookup_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_map_lookup_elem(fd, key);
}

long bpftime_map_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags)
{
	return shm_holder.global_shared_memory.bpf_update_elem(fd, key, value,
							       flags);
}

long bpftime_map_delete_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_delete_elem(fd, key);
}
int bpftime_map_get_next_key(int fd, const void *key, void *next_key)
{
	return shm_holder.global_shared_memory.bpf_map_get_next_key(fd, key,
								    next_key);
}

int bpftime_uprobe_create(int pid, const char *name, uint64_t offset,
			  bool retprobe, size_t ref_ctr_off)
{
	return shm_holder.global_shared_memory.add_uprobe(
		pid, name, offset, retprobe, ref_ctr_off);
}

int bpftime_tracepoint_create(int pid, int32_t tp_id)
{
	return shm_holder.global_shared_memory.add_tracepoint(pid, tp_id);
}

int bpftime_attach_enable(int fd)
{
	return shm_holder.global_shared_memory.attach_enable(fd);
}

int bpftime_attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	return shm_holder.global_shared_memory.attach_perf_to_bpf(perf_fd,
								  bpf_fd);
}
int bpftime_add_ringbuf_fd_to_epoll(int ringbuf_fd, int epoll_fd)
{
	return shm_holder.global_shared_memory.add_ringbuf_to_epoll(ringbuf_fd,
								    epoll_fd);
}
int bpftime_epoll_create()
{
	return shm_holder.global_shared_memory.epoll_create();
}
void bpftime_close(int fd)
{
	shm_holder.global_shared_memory.close_fd(fd);
}
int bpftime_map_get_info(int fd, bpftime::bpf_map_attr *out_attr,
			 const char **out_name, int *type)
{
	if (!shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_map_handler>(
		shm_holder.global_shared_memory.get_handler(fd));
	if (out_attr) {
		*out_attr = handler.attr;
	}
	if (out_name) {
		*out_name = handler.name.c_str();
	}
	if (type) {
		*type = handler.type;
	}
	return 0;
}

extern "C" uint64_t map_ptr_by_fd(uint32_t fd)
{
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	// Use a convenient way to represent a pointer
	return ((uint64_t)fd << 32) | 0xffffffff;
}

extern "C" uint64_t map_val(uint64_t map_ptr)
{
	int fd = (int)(map_ptr >> 32);
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		return 0;
	}
	auto &handler = std::get<bpftime::bpf_map_handler>(
		shm_holder.global_shared_memory.get_handler(fd));
	auto size = handler.attr.key_size;
	std::vector<char> key(size);
	int res = handler.bpf_map_get_next_key(nullptr, key.data());
	if (res < 0) {
		errno = ENOENT;
		return 0;
	}
	return (uint64_t)handler.map_lookup_elem(key.data());
}
