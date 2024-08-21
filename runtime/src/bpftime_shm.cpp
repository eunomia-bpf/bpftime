/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_prog.hpp"
#include "handler/epoll_handler.hpp"
#include "handler/map_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <csignal>
#include <cstddef>
#include <exception>
#include <signal.h>
#include <cerrno>
#include <errno.h>
#include <bpftime_shm_internal.hpp>
#if __linux__
#include <sys/epoll.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include <thread>
#include <chrono>
#include <variant>

#ifdef __APPLE__
// Custom implementation for sigtimedwait
int sigtimedwait(const sigset_t *set, siginfo_t *info,
		 const struct timespec *timeout)
{
	struct timespec start, now;
	clock_gettime(CLOCK_REALTIME, &start);
	int sig;

	while (true) {
		// Try to wait for a signal
		if (sigwait(set, &sig) == 0) {
			if (info != nullptr) {
				memset(info, 0, sizeof(*info));
				info->si_signo = sig;
			}
			return sig;
		}

		// Check if the timeout has expired
		clock_gettime(CLOCK_REALTIME, &now);
		if ((now.tv_sec - start.tv_sec) > timeout->tv_sec ||
		    ((now.tv_sec - start.tv_sec) == timeout->tv_sec &&
		     (now.tv_nsec - start.tv_nsec) > timeout->tv_nsec)) {
			errno = EAGAIN;
			return -1;
		}

		// Sleep for a short time before retrying
		usleep(1000); // Sleep for 1ms before retrying
	}
}
#endif

using namespace bpftime;

int bpftime_find_minimal_unused_fd()
{
	return shm_holder.global_shared_memory.find_minimal_unused_fd();
}

int bpftime_link_create(int fd, struct bpf_link_create_args *args)
{
	return shm_holder.global_shared_memory.add_bpf_link(fd, args);
}

int bpftime_progs_create(int fd, const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
{
	return shm_holder.global_shared_memory.add_bpf_prog(
		fd, insn, insn_cnt, prog_name, prog_type);
}

int bpftime_maps_create(int fd, const char *name, bpftime::bpf_map_attr attr)
{
	return shm_holder.global_shared_memory.add_bpf_map(fd, name, attr);
}

uint32_t bpftime_map_value_size_from_syscall(int fd)
{
	return shm_holder.global_shared_memory.bpf_map_value_size(fd);
}

int bpftime_helper_map_get_next_key(int fd, const void *key, void *next_key)
{
	try {
		return shm_holder.global_shared_memory.bpf_map_get_next_key(
			fd, key, next_key, false);
	} catch (std::exception &ex) {
		SPDLOG_ERROR(
			"Exception happened when performing map get next key (from helper): {}",
			ex.what());
		return -1;
	}
}

const void *bpftime_map_lookup_elem(int fd, const void *key)
{
	try {
		return shm_holder.global_shared_memory.bpf_map_lookup_elem(
			fd, key, true);
	} catch (std::exception &ex) {
		SPDLOG_ERROR(
			"Exception happened when performing map lookup elem: {}",
			ex.what());
		return nullptr;
	}
}

long bpftime_map_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags)
{
	try {
		return shm_holder.global_shared_memory.bpf_map_update_elem(
			fd, key, value, flags, true);
	} catch (std::exception &ex) {
		SPDLOG_ERROR(
			"Exception happened when performing map update: {}",
			ex.what());
		return -1;
	}
}

long bpftime_map_delete_elem(int fd, const void *key)
{
	try {
		return shm_holder.global_shared_memory.bpf_delete_elem(fd, key,
								       true);
	} catch (std::exception &ex) {
		SPDLOG_ERROR(
			"Exception happened when performing map delete elem: {}",
			ex.what());
		return -1;
	}
}

int bpftime_map_get_next_key(int fd, const void *key, void *next_key)
{
	try {
		return shm_holder.global_shared_memory.bpf_map_get_next_key(
			fd, key, next_key, true);
	} catch (std::exception &ex) {
		SPDLOG_ERROR(
			"Exception happened when performing map get next key: {}",
			ex.what());
		return -1;
	}
}

int bpftime_uprobe_create(int fd, int pid, const char *name, uint64_t offset,
			  bool retprobe, size_t ref_ctr_off)
{
	return shm_holder.global_shared_memory.add_uprobe(
		fd, pid, name, offset, retprobe, ref_ctr_off);
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
								  bpf_fd, {});
}

int bpftime_attach_perf_to_bpf_with_cookie(int perf_fd, int bpf_fd,
					   uint64_t cookie)
{
	return shm_holder.global_shared_memory.attach_perf_to_bpf(
		perf_fd, bpf_fd, cookie);
}

int bpftime_get_current_thread_cookie(uint64_t *out)
{
	if (current_thread_bpf_cookie.has_value()) {
		*out = *current_thread_bpf_cookie;
		return 1;
	}
	return 0;
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
			 const char **out_name, bpftime::bpf_map_type *type)
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

void bpftime_protect_enable()
{
#if BPFTIME_ENABLE_MPK
	return shm_holder.global_shared_memory.enable_mpk();
#endif
}

void bpftime_protect_disable()
{
#if BPFTIME_ENABLE_MPK
	return shm_holder.global_shared_memory.disable_mpk();
#endif
}

int bpftime_is_map_fd(int fd)
{
	return shm_holder.global_shared_memory.is_map_fd(fd);
}

int bpftime_is_perf_event_fd(int fd)
{
	return shm_holder.global_shared_memory.is_perf_event_handler_fd(fd);
}

int bpftime_is_prog_fd(int fd)
{
	return shm_holder.global_shared_memory.is_prog_fd(fd);
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
		SPDLOG_ERROR("Expected fd {} to be ringbuf map fd ",
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
		SPDLOG_ERROR("Expected fd {} to be ringbuf map fd ",
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
		SPDLOG_ERROR("Expected fd {} to be ringbuf map fd ", fd);
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
		SPDLOG_ERROR("Expected fd {} to be ringbuf map fd ", fd);
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
		SPDLOG_ERROR("Expected {} to be an epoll fd", fd);
		return -1;
	}
	using namespace std::chrono;
	auto &epoll_inst =
		std::get<epoll_handler>(shm.get_manager()->get_handler(fd));
	auto start_time = high_resolution_clock::now();
	int next_id = 0;
	sigset_t orig_sigset;
	sigset_t to_block;
	sigemptyset(&to_block);
	sigaddset(&to_block, SIGINT);
	sigaddset(&to_block, SIGTERM);

	// Block the delivery of some signals, so we would be able to catch
	// them when sleeping
	if (int err = sigprocmask(SIG_BLOCK, &to_block, &orig_sigset);
	    err == -1) {
		SPDLOG_ERROR(
			"sigprocmask failed to block sigint & sigterm, errno={}. this SHOULD NOT HAPPEN",
			errno);
		errno = EINVAL;
		return -1;
	}
	// timeout for waiting..
	timespec ts{ .tv_sec = 0, .tv_nsec = 1000 * 1000 };
	bool failed_with_intr = false;
	while (next_id < max_evt) {
		auto now_time = high_resolution_clock::now();
		auto elasped =
			duration_cast<milliseconds>(now_time - start_time);
		if (timeout && elasped.count() > timeout) {
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
		if (next_id > 0) {
			// According to man epoll_wait(2), epoll_wait can't be
			// interrupted once at least one event was received
			std::this_thread::sleep_for(milliseconds(1));
		} else {
			// Nothing has been received, so allow the interruption
			// of epoll_wait
			// First, unblock the signals
			sigprocmask(SIG_UNBLOCK, &to_block, nullptr);
			siginfo_t sig_info;
			// Second, wait for interruptable signals
			if (int sig = sigtimedwait(&to_block, &sig_info, &ts);
			    sig > 0) {
				SPDLOG_DEBUG(
					"epoll_wait interrupted by signal {}",
					sig);
				// Invoke the original signal handler
				struct sigaction act;
				sigaction(sig, nullptr, &act);
				if ((act.sa_flags & SA_SIGINFO) &&
				    act.sa_sigaction) {
					act.sa_sigaction(sig, &sig_info,
							 nullptr);
				} else if (auto f = act.sa_handler) {
					f(sig);
				}
				failed_with_intr = true;
				break;
			}
			// If not catched, just block them again
			sigprocmask(SIG_BLOCK, &to_block, nullptr);
		}
	}
	// Restore the original sigmask
	sigprocmask(SIG_SETMASK, &orig_sigset, nullptr);
	if (failed_with_intr) {
		errno = EINTR;
		return -1;
	}
	return next_id;
}

int bpftime_add_software_perf_event(int cpu, int32_t sample_type,
				    int64_t config)
{
	auto &shm = shm_holder.global_shared_memory;
	return shm.add_software_perf_event(cpu, sample_type, config);
}

int bpftime_add_ureplace_or_override(int fd, int pid, const char *name,
				     uint64_t offset, bool is_replace)
{
	auto &shm = shm_holder.global_shared_memory;
	return shm.add_uprobe_override(fd, pid, name, offset, is_replace);
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
		SPDLOG_ERROR("Expected fd {} to be a perf event handler", fd);
		errno = EINVAL;
		return -1;
	}
	auto &handler = std::get<bpf_perf_event_handler>(shm.get_handler(fd));
	if (std::holds_alternative<software_perf_event_shared_ptr>(
		    handler.data)) {
		SPDLOG_DEBUG("Perf out value to fd {}, sz {}", fd, sz);
		return std::get<software_perf_event_shared_ptr>(handler.data)
			->output_data(buf, sz);
	} else {
		SPDLOG_ERROR(
			"Expected perf event handler {} to be a software perf event handler",
			fd);
		errno = ENOTSUP;
		return -1;
	}
}

#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
int bpftime_shared_perf_event_output(int map_fd, const void *buf, size_t sz)
{
	SPDLOG_DEBUG("Output data into shared perf event array fd {}", map_fd);
	auto &shm = shm_holder.global_shared_memory;
	if (!shm.is_shared_perf_event_array_map_fd(map_fd)) {
		SPDLOG_ERROR("Expected fd {} to be a shared perf event array",
			     map_fd);
		errno = EINVAL;
		return -1;
	}
	auto &map_handler = std::get<bpf_map_handler>(shm.get_handler(map_fd));
	if (auto p = map_handler.try_get_shared_perf_event_array_map_impl();
	    p.has_value()) {
		int err = p.value()->output_data_into_kernel(buf, sz);
		if (err < 0) {
			errno = -err;
			return -1;
		}
		return 0;
	} else {
		SPDLOG_ERROR(
			"Expected map {} to be a shared perf event array map",
			map_fd);
		errno = EINVAL;
		return -1;
	}
}
#endif

int bpftime_is_prog_array(int fd)
{
	return shm_holder.global_shared_memory.is_prog_array_map_fd(fd);
}

const uint64_t INVALID_MAP_PTR = ((uint64_t)0 - 1);

extern "C" uint64_t map_ptr_by_fd(uint32_t fd)
{
	SPDLOG_DEBUG("Call map_ptr_by_fd with fd={}", fd);
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		errno = ENOENT;
		SPDLOG_ERROR("Expected fd {} to be a map fd (map_ptr_by_fd)",
			     fd);
		// Here we just ignore the wrong maps
		return INVALID_MAP_PTR;
	}
	// Use a convenient way to represent a pointer
	return ((uint64_t)fd << 32) | 0xffffffff;
}

extern "C" uint64_t map_val(uint64_t map_ptr)
{
	SPDLOG_DEBUG("Call map_val with map_ptr={:x}", map_ptr);
	int fd = (int)(map_ptr >> 32);
	if (!shm_holder.global_shared_memory.get_manager() ||
	    !shm_holder.global_shared_memory.is_map_fd(fd)) {
		SPDLOG_ERROR("Expected fd {} to be a map fd (map_val call)",
			     fd);
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

int bpftime_add_custom_perf_event(int type, const char *attach_argument)
{
	return shm_holder.global_shared_memory.add_custom_perf_event(
		type, attach_argument);
}

int bpftime_poll_from_ringbuf(int rb_fd, void *ctx,
			      int (*cb)(void *, void *, size_t))
{
	auto &shm = shm_holder.global_shared_memory;
	if (auto ret = shm.try_get_ringbuf_map_impl(rb_fd); ret.has_value()) {
		auto impl = ret.value();
		return impl->create_impl_shared_ptr()->fetch_data(
			[=](void *buf, int sz) { return cb(ctx, buf, sz); });
	} else {
		errno = EINVAL;
		SPDLOG_ERROR("Expected fd {} to be ringbuf map fd ", rb_fd);
		return -EINVAL;
	}
}
