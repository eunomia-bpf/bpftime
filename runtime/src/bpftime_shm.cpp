#include "handler/epoll_handler.hpp"
#include "handler/map_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <errno.h>
#include <bpftime_shm_internal.hpp>
#include <sys/epoll.h>
#include <thread>
#include <chrono>
#include <variant>

using namespace bpftime;

int bpftime_link_create(int fd, int prog_fd, int target_fd)
{
	return shm_holder.global_shared_memory.add_bpf_link(fd, prog_fd, target_fd);
}

int bpftime_progs_create(int fd, const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
{
	return shm_holder.global_shared_memory.add_bpf_prog(fd, 
		insn, insn_cnt, prog_name, prog_type);
}

int bpftime_maps_create(int fd, const char *name, bpftime::bpf_map_attr attr)
{
	return shm_holder.global_shared_memory.add_bpf_map(fd, name, attr);
}
uint32_t bpftime_map_value_size_from_syscall(int fd)
{
	return shm_holder.global_shared_memory.bpf_map_value_size(fd);
}

const void *bpftime_helper_map_lookup_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_map_lookup_elem(fd, key,
								   false);
}

long bpftime_helper_map_update_elem(int fd, const void *key,
					 const void *value, uint64_t flags)
{
	return shm_holder.global_shared_memory.bpf_map_update_elem(
		fd, key, value, flags, false);
}

long bpftime_helper_map_delete_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_delete_elem(fd, key, false);
}
int bpftime_helper_map_get_next_key(int fd, const void *key,
					 void *next_key)
{
	return shm_holder.global_shared_memory.bpf_map_get_next_key(
		fd, key, next_key, false);
}

const void *bpftime_map_lookup_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_map_lookup_elem(fd, key,
								   true);
}

long bpftime_map_update_elem(int fd, const void *key,
					  const void *value, uint64_t flags)
{
	return shm_holder.global_shared_memory.bpf_map_update_elem(
		fd, key, value, flags, true);
}

long bpftime_map_delete_elem(int fd, const void *key)
{
	return shm_holder.global_shared_memory.bpf_delete_elem(fd, key, true);
}
int bpftime_map_get_next_key(int fd, const void *key,
					  void *next_key)
{
	return shm_holder.global_shared_memory.bpf_map_get_next_key(
		fd, key, next_key, true);
}

int bpftime_uprobe_create(int fd, int pid, const char *name, uint64_t offset,
			  bool retprobe, size_t ref_ctr_off)
{
	return shm_holder.global_shared_memory.add_uprobe(fd, 
		pid, name, offset, retprobe, ref_ctr_off);
}

int bpftime_tracepoint_create(int fd, int pid, int32_t tp_id)
{
	return shm_holder.global_shared_memory.add_tracepoint(fd, pid, tp_id);
}

int bpftime_perf_event_enable(int fd)
{
	return shm_holder.global_shared_memory.perf_event_enable(fd);
}
int bpftime_perf_event_disable(int fd)
{
	return shm_holder.global_shared_memory.perf_event_disable(fd);
}

int bpftime_attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	return shm_holder.global_shared_memory.attach_perf_to_bpf(perf_fd,
								  bpf_fd);
}
int bpftime_add_ringbuf_fd_to_epoll(int ringbuf_fd, int epoll_fd,
				    epoll_data_t extra_data)
{
	return shm_holder.global_shared_memory.add_ringbuf_to_epoll(
		ringbuf_fd, epoll_fd, extra_data);
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
			 const char **out_name, bpf_map_type *type)
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

int bpftime_is_ringbuf_map(int fd)
{
	return shm_holder.global_shared_memory.is_ringbuf_map_fd(fd);
}

int bpftime_is_array_map(int fd)
{
	return shm_holder.global_shared_memory.is_array_map_fd(fd);
}

void *bpftime_get_array_map_raw_data(int fd)
{
	if (auto array_impl =
		    shm_holder.global_shared_memory.try_get_array_map_impl(fd);
	    array_impl.has_value()) {
		return array_impl.value()->get_raw_data();
	} else {
		errno = EINVAL;

		return nullptr;
	}
}

void *bpftime_get_ringbuf_consumer_page(int ringbuf_fd)
{
	auto &shm = shm_holder.global_shared_memory;
	if (auto ret = shm.try_get_ringbuf_map_impl(ringbuf_fd);
	    ret.has_value()) {
		return ret.value()->get_consumer_page();
	} else {
		errno = EINVAL;
		spdlog::error("Expected fd {} to be ringbuf map fd ",
			      ringbuf_fd);
		return nullptr;
	}
}

void *bpftime_get_ringbuf_producer_page(int ringbuf_fd)
{
	auto &shm = shm_holder.global_shared_memory;
	if (auto ret = shm.try_get_ringbuf_map_impl(ringbuf_fd);
	    ret.has_value()) {
		return ret.value()->get_producer_page();
	} else {
		errno = EINVAL;
		spdlog::error("Expected fd {} to be ringbuf map fd ",
			      ringbuf_fd);
		return nullptr;
	}
}

void *bpftime_ringbuf_reserve(int fd, uint64_t size)
{
	auto &shm = shm_holder.global_shared_memory;
	if (auto ret = shm.try_get_ringbuf_map_impl(fd); ret.has_value()) {
		auto impl = ret.value();
		return impl->reserve(size, fd);
	} else {
		errno = EINVAL;
		spdlog::error("Expected fd {} to be ringbuf map fd ", fd);
		return nullptr;
	}
}

void bpftime_ringbuf_submit(int fd, void *data, int discard)
{
	auto &shm = shm_holder.global_shared_memory;
	if (auto ret = shm.try_get_ringbuf_map_impl(fd); ret.has_value()) {
		auto impl = ret.value();
		impl->submit(data, discard);
	} else {
		errno = EINVAL;
		spdlog::error("Expected fd {} to be ringbuf map fd ", fd);
	}
}

int bpftime_is_epoll_handler(int fd)
{
	return shm_holder.global_shared_memory.is_epoll_fd(fd);
}

int bpftime_epoll_wait(int fd, struct epoll_event *out_evts, int max_evt,
		       int timeout)
{
	auto &shm = shm_holder.global_shared_memory;
	if (!shm.is_epoll_fd(fd)) {
		errno = EINVAL;
		spdlog::error("Expected {} to be an epoll fd", fd);
		return -1;
	}
	using namespace std::chrono;
	auto &epoll_inst =
		std::get<epoll_handler>(shm.get_manager()->get_handler(fd));
	auto start_time = high_resolution_clock::now();
	int next_id = 0;
	while (next_id < max_evt) {
		auto now_time = high_resolution_clock::now();
		auto elasped =
			duration_cast<milliseconds>(now_time - start_time);
		if (elasped.count() > timeout) {
			break;
		}
		for (const auto &p : epoll_inst.files) {
			if (std::holds_alternative<software_perf_event_weak_ptr>(
				    p.file)) {
				if (auto ptr =
					    std::get<software_perf_event_weak_ptr>(
						    p.file)
						    .lock();
				    ptr) {
					if (ptr->has_data() &&
					    next_id < max_evt) {
						out_evts[next_id++] =
							epoll_event{
								.events =
									EPOLLIN,
								.data = p.data
							};
					}
				}
			} else if (std::holds_alternative<ringbuf_weak_ptr>(
					   p.file)) {
				if (auto ptr =
					    std::get<ringbuf_weak_ptr>(p.file)
						    .lock();
				    ptr) {
					if (ptr->has_data() &&
					    next_id < max_evt) {
						out_evts[next_id++] =
							epoll_event{
								.events =
									EPOLLIN,
								.data = p.data
							};
					}
				}
			}
		}
		std::this_thread::sleep_for(milliseconds(1));
	}
	return next_id;
}

int bpftime_add_software_perf_event(int cpu, int32_t sample_type,
				    int64_t config)
{
	auto &shm = shm_holder.global_shared_memory;
	return shm.add_software_perf_event(cpu, sample_type, config);
}

int bpftime_add_software_perf_event_fd_to_epoll(int swpe_fd, int epoll_fd,
						epoll_data_t extra_data)
{
	return shm_holder.global_shared_memory.add_software_perf_event_to_epoll(
		swpe_fd, epoll_fd, extra_data);
}

int bpftime_is_software_perf_event(int fd)
{
	return shm_holder.global_shared_memory.is_software_perf_event_handler_fd(
		fd);
}

void *bpftime_get_software_perf_event_raw_buffer(int fd, size_t expected_size)
{
	return shm_holder.global_shared_memory
		.get_software_perf_event_raw_buffer(fd, expected_size)
		.value_or(nullptr);
}

int bpftime_perf_event_output(int fd, const void *buf, size_t sz)
{
	auto &shm = shm_holder.global_shared_memory;
	if (!shm.is_perf_event_handler_fd(fd)) {
		spdlog::error("Expected fd {} to be a perf event handler", fd);
		errno = EINVAL;
		return -1;
	}
	auto &handler = std::get<bpf_perf_event_handler>(shm.get_handler(fd));
	if (handler.sw_perf.has_value()) {
		spdlog::debug("Perf out value to fd {}, sz {}", fd, sz);
		return handler.sw_perf.value()->output_data(buf, sz);
	} else {
		spdlog::error(
			"Expected perf event handler {} to be a software perf event handler",
			fd);
		errno = ENOTSUP;
		return -1;
	}
}

extern "C" uint64_t map_ptr_by_fd(uint32_t fd)
{
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		spdlog::error("Expected fd {} to be a map fd", fd);
		// Here we just ignore the wrong maps
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
		spdlog::error("Expected fd {} to be a map fd", fd);
		// here we just ignore the wrong maps
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
