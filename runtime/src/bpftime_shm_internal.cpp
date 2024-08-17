/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm.hpp"
#include <ebpf-vm.h>
#include "handler/epoll_handler.hpp"
#include "handler/handler_manager.hpp"
#include "handler/link_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_shm_internal.hpp>
#include <cerrno>
#include <cstdio>
#if __linux__
#include <sys/epoll.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include <unistd.h>
#include <variant>
#include <sys/mman.h>

static bool global_shm_initialized = false;

extern "C" void bpftime_initialize_global_shm(bpftime::shm_open_type type)
{
	using namespace bpftime;
	// Use placement new, which will not allocate memory, but just
	// call the constructor
	new (&shm_holder.global_shared_memory) bpftime_shm(type);
	global_shm_initialized = true;
	SPDLOG_INFO("Global shm initialized");
}

extern "C" void bpftime_destroy_global_shm()
{
	using namespace bpftime;
	if (global_shm_initialized) {
		// SPDLOG_INFO("Global shm destructed");
		shm_holder.global_shared_memory.~bpftime_shm();
		// Why not spdlog? because global variables that spdlog used
		// were already destroyed..
#ifdef DEBUG
		fprintf(stderr, "INFO [%d]: Global shm destructed\n",
			(int)getpid());
#endif
	}
}

extern "C" void bpftime_remove_global_shm()
{
	using namespace bpftime;
	if (boost::interprocess::shared_memory_object::remove(
		    get_global_shm_name()) != false) {
		SPDLOG_INFO("Global shm removed");
	}
}

static __attribute__((destructor(65535))) void __destruct_shm()
{
	// This usually indicates that the living shared memory object is used
	// by an agent instance
	if (bpftime::shm_holder.global_shared_memory.get_open_type() ==
	    bpftime::shm_open_type::SHM_OPEN_ONLY) {
		// Try our best to remove the current pid from alive agent's set
		int self_pid = getpid();
		// It doesn't matter if the current pid is not in the set
		bpftime::shm_holder.global_shared_memory
			.remove_pid_from_alive_agent_set(self_pid);
	}

	bpftime_destroy_global_shm();
}

namespace bpftime
{

bpftime_shm_holder shm_holder;

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

const void *bpftime_shm::bpf_map_lookup_elem(int fd, const void *key,
					     bool from_syscall) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return nullptr;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_lookup_elem(key, from_syscall);
}

long bpftime_shm::bpf_map_update_elem(int fd, const void *key,
				      const void *value, uint64_t flags,
				      bool from_syscall) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_update_elem(key, value, flags, from_syscall);
}

long bpftime_shm::bpf_delete_elem(int fd, const void *key,
				  bool from_syscall) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_delete_elem(key, from_syscall);
}

int bpftime_shm::bpf_map_get_next_key(int fd, const void *key, void *next_key,
				      bool from_syscall) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.bpf_map_get_next_key(key, next_key, from_syscall);
}

int bpftime_shm::add_uprobe(int fd, int pid, const char *name, uint64_t offset,
			    bool retprobe, size_t ref_ctr_off)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	SPDLOG_DEBUG("Set fd {} to uprobe, pid={}, name={}, offset={}", fd, pid,
		     name, offset);
	return manager->set_handler(
		fd,
		bpftime::bpf_perf_event_handler{ retprobe, offset, pid, name,
						 ref_ctr_off, segment },
		segment);
}

int bpftime_shm::add_uprobe_override(int fd, int pid, const char *name,
				     uint64_t offset, bool is_replace)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	SPDLOG_DEBUG("Set fd {} to ureplace, pid={}, name={}, offset={}", fd,
		     pid, name, offset);
	if (is_replace) {
		return manager->set_handler(
			fd,
			bpf_perf_event_handler(
				bpf_event_type::BPF_TYPE_UREPLACE, offset, pid,
				name, segment, true),
			segment);
	} else {
		return manager->set_handler(
			fd,
			bpf_perf_event_handler(
				bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE,
				offset, pid, name, segment, true),
			segment);
	}
}

int bpftime_shm::add_tracepoint(int fd, int pid, int32_t tracepoint_id)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	return manager->set_handler(
		fd,
		bpftime::bpf_perf_event_handler(pid, tracepoint_id, segment),
		segment);
}

int bpftime_shm::add_software_perf_event(int cpu, int32_t sample_type,
					 int64_t config)
{
	return add_software_perf_event(open_fake_fd(), cpu, sample_type,
				       config);
}

int bpftime_shm::add_software_perf_event(int fd, int cpu, int32_t sample_type,
					 int64_t config)
{
	return manager->set_handler(fd,
				    bpftime::bpf_perf_event_handler(
					    cpu, sample_type, config, segment),
				    segment);
	return fd;
}

int bpftime_shm::attach_perf_to_bpf(int perf_fd, int bpf_fd,
				    std::optional<uint64_t> cookie)
{
	if (!is_perf_fd(perf_fd)) {
		SPDLOG_ERROR("Fd {} is not a perf fd", perf_fd);
		errno = ENOENT;
		return -1;
	}
	return add_bpf_prog_attach_target(perf_fd, bpf_fd, cookie);
}

int bpftime_shm::add_bpf_prog_attach_target(int perf_fd, int bpf_fd,
					    std::optional<uint64_t> cookie)
{
	SPDLOG_DEBUG("Try attaching prog fd {} to perf fd {}, with cookie = {}",
		     bpf_fd, perf_fd, cookie.has_value());
	if (cookie) {
		SPDLOG_DEBUG("With cookie: {}", *cookie);
	}

	if (!is_prog_fd(bpf_fd)) {
		SPDLOG_ERROR("Fd {} is not a prog fd", bpf_fd);
		errno = ENOENT;
		return -1;
	}
	int next_id = open_fake_fd();
	if (next_id < 0) {
		SPDLOG_ERROR("Unable to find an available id: {}", next_id);
		return -ENOSPC;
	}
	manager->set_handler(next_id, bpf_link_handler(bpf_fd, perf_fd, cookie),
			     segment);
	return next_id;
}

int bpftime_shm::perf_event_enable(int fd) const
{
	if (!is_perf_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_perf_event_handler>(
		manager->get_handler(fd));
	return handler.enable();
}

int bpftime_shm::perf_event_disable(int fd) const
{
	if (!is_perf_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_perf_event_handler>(
		manager->get_handler(fd));
	return handler.disable();
}

int bpftime_shm::add_software_perf_event_to_epoll(int swpe_fd, int epoll_fd,
						  epoll_data_t extra_data)
{
	if (!is_epoll_fd(epoll_fd)) {
		SPDLOG_ERROR("Fd {} is expected to be an epoll fd", epoll_fd);
		errno = EINVAL;
		return -1;
	}
	auto &epoll_inst =
		std::get<epoll_handler>(manager->get_handler(epoll_fd));
	if (!is_software_perf_event_handler_fd(swpe_fd)) {
		SPDLOG_ERROR(
			"Fd {} is expected to be an software perf event handler",
			swpe_fd);
		errno = EINVAL;
		return -1;
	}
	auto &perf_handler =
		std::get<bpf_perf_event_handler>(manager->get_handler(swpe_fd));
	if (perf_handler.type != (int)bpf_event_type::PERF_TYPE_SOFTWARE) {
		SPDLOG_ERROR(
			"Expected perf fd {} to be a software perf event instance",
			swpe_fd);
		errno = EINVAL;
		return -1;
	}
	if (auto ptr = perf_handler.try_get_software_perf_data_weak_ptr();
	    ptr.has_value()) {
		epoll_inst.files.emplace_back(ptr.value(), extra_data);
		return 0;
	} else {
		SPDLOG_ERROR(
			"Expected perf handler {} to have software perf event data",
			swpe_fd);
		errno = EINVAL;
		return -1;
	}
}
int bpftime_shm::add_ringbuf_to_epoll(int ringbuf_fd, int epoll_fd,
				      epoll_data_t extra_data)
{
	if (!is_epoll_fd(epoll_fd)) {
		SPDLOG_ERROR("Fd {} is expected to be an epoll fd", epoll_fd);
		errno = EINVAL;
		return -1;
	}
	auto &epoll_inst =
		std::get<epoll_handler>(manager->get_handler(epoll_fd));

	if (!is_map_fd(ringbuf_fd)) {
		SPDLOG_ERROR("Fd {} is expected to be an map fd", ringbuf_fd);
		errno = EINVAL;
		return -1;
	}
	auto &map_inst =
		std::get<bpf_map_handler>(manager->get_handler(ringbuf_fd));

	auto ringbuf_map_impl = map_inst.try_get_ringbuf_map_impl();
	if (ringbuf_map_impl.has_value(); auto val = ringbuf_map_impl.value()) {
		epoll_inst.files.emplace_back(val->create_impl_weak_ptr(),
					      extra_data);
		SPDLOG_DEBUG("Ringbuf {} added to epoll {}", ringbuf_fd,
			     epoll_fd);
		return 0;
	} else {
		errno = EINVAL;
		SPDLOG_ERROR("Map fd {} is expected to be an ringbuf map",
			     ringbuf_fd);
		return -1;
	}
}
int bpftime_shm::epoll_create()
{
	int fd = open_fake_fd();
	if (manager->is_allocated(fd)) {
		SPDLOG_ERROR(
			"Creating epoll instance, but fd {} is already occupied",
			fd);
		return -1;
	}
	fd = manager->set_handler(fd, bpftime::epoll_handler(segment), segment);
	SPDLOG_DEBUG("Epoll instance created: fd={}", fd);
	return fd;
}

const handler_variant &bpftime_shm::get_handler(int fd) const
{
	return manager->get_handler(fd);
}
bool bpftime_shm::is_epoll_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		SPDLOG_ERROR("Invalid fd: {}", fd);
		return false;
	}
	const auto &handler = manager->get_handler(fd);
	return std::holds_alternative<bpftime::epoll_handler>(handler);
}

bool bpftime_shm::is_map_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	const auto &handler = manager->get_handler(fd);
	return std::holds_alternative<bpftime::bpf_map_handler>(handler);
}
bool bpftime_shm::is_ringbuf_map_fd(int fd) const
{
	if (!is_map_fd(fd))
		return false;
	auto &map_impl = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_impl.type == bpf_map_type::BPF_MAP_TYPE_RINGBUF;
}
bool bpftime_shm::is_shared_perf_event_array_map_fd(int fd) const
{
	if (!is_map_fd(fd))
		return false;
	auto &map_impl = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_impl.type ==
	       bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY;
}
bool bpftime_shm::is_array_map_fd(int fd) const
{
	if (!is_map_fd(fd))
		return false;
	auto &map_impl = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_impl.type == bpf_map_type::BPF_MAP_TYPE_ARRAY;
}

bool bpftime_shm::is_prog_array_map_fd(int fd) const
{
	if (!is_map_fd(fd))
		return false;
	auto &map_impl = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_impl.type == bpf_map_type::BPF_MAP_TYPE_PROG_ARRAY;
}
std::optional<ringbuf_map_impl *>
bpftime_shm::try_get_ringbuf_map_impl(int fd) const
{
	if (!is_ringbuf_map_fd(fd)) {
		SPDLOG_ERROR("Expected fd {} to be an ringbuf map fd", fd);
		return {};
	}
	auto &map_handler = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_handler.try_get_ringbuf_map_impl();
}

std::optional<array_map_impl *>
bpftime_shm::try_get_array_map_impl(int fd) const
{
	if (!is_array_map_fd(fd)) {
		SPDLOG_ERROR("Expected fd {} to be an array map fd", fd);
		return {};
	}
	auto &map_handler = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_handler.try_get_array_map_impl();
}
bool bpftime_shm::is_prog_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	const auto &handler = manager->get_handler(fd);
	return std::holds_alternative<bpftime::bpf_prog_handler>(handler);
}

bool bpftime_shm::is_perf_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	const auto &handler = manager->get_handler(fd);
	return std::holds_alternative<bpftime::bpf_perf_event_handler>(handler);
}

int bpftime_shm::open_fake_fd()
{
	int fd = open("/dev/null", O_RDONLY);
	int cnt = 5;
	while (fd <= 2 && fd >= 0 && --cnt > 0) {
		fd = dup(fd);
	}
	return fd;
}

// handle bpf commands to load a bpf program
int bpftime_shm::add_bpf_prog(int fd, const ebpf_inst *insn, size_t insn_cnt,
			      const char *prog_name, int prog_type)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	SPDLOG_DEBUG(
		"Set handler fd {} to bpf_prog_handler, name {}, prog_type {}, insn_cnt {}",
		fd, prog_name, prog_type, insn_cnt);
	return manager->set_handler(
		fd,
		bpftime::bpf_prog_handler(segment, insn, insn_cnt, prog_name,
					  prog_type),
		segment);
}

// add a bpf link fd
int bpftime_shm::add_bpf_link(int fd, struct bpf_link_create_args *args)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	if (!is_prog_fd(args->prog_fd) || !args) {
		errno = EBADF;
		return -1;
	}
	return manager->set_handler(fd, bpftime::bpf_link_handler(*args),
				    segment);
}

void bpftime_shm::close_fd(int fd)
{
	if (manager) {
		manager->clear_id_at(fd, segment);
	}
}

#if BPFTIME_ENABLE_MPK
void bpftime_shm::enable_mpk()
{
	if (manager == nullptr || !is_mpk_init) {
		return;
	}
	if (pkey_set(pkey, PKEY_DISABLE_WRITE) == -1) {
		SPDLOG_ERROR("pkey_set read only failed");
	}
}

void bpftime_shm::disable_mpk()
{
	if (manager == nullptr || !is_mpk_init) {
		return;
	}
	if (pkey_set(pkey, 0) == -1) {
		SPDLOG_ERROR("pkey_set disable failed");
	}
}
#endif

bool bpftime_shm::is_exist_fake_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	return manager->is_allocated(fd);
}

bpftime_shm::bpftime_shm(const char *shm_name, shm_open_type type)
	: open_type(type)
{
	// Get the config from env because the shared memory is not initialized
	size_t memory_size = get_agent_config_from_env().shm_memory_size;
	if (type == shm_open_type::SHM_OPEN_ONLY) {
		SPDLOG_DEBUG("start: bpftime_shm for client setup");
		// open the shm
		segment = boost::interprocess::managed_shared_memory(
			boost::interprocess::open_only, shm_name);
		manager = segment.find<bpftime::handler_manager>(
					 bpftime::DEFAULT_GLOBAL_HANDLER_NAME)
				  .first;
		syscall_installed_pids =
			segment.find<syscall_pid_set>(
				       DEFAULT_SYSCALL_PID_SET_NAME)
				.first;
		agent_config =
			segment.find<struct agent_config>(
				       bpftime::DEFAULT_AGENT_CONFIG_NAME)
				.first;

		injected_pids =
			segment.find<alive_agent_pids>(
				       bpftime::DEFAULT_ALIVE_AGENT_PIDS_NAME)
				.first;
		SPDLOG_DEBUG("done: bpftime_shm for client setup");
	} else if (type == shm_open_type::SHM_CREATE_OR_OPEN) {
		SPDLOG_DEBUG(
			"start: bpftime_shm for create or open setup for memory size {}",
			memory_size);
		segment = boost::interprocess::managed_shared_memory(
			boost::interprocess::open_or_create,
			// Allocate 20M bytes of memory by default
			shm_name, memory_size << 20);

		manager = segment.find_or_construct<bpftime::handler_manager>(
			bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
		SPDLOG_DEBUG("done: bpftime_shm for server setup: manager");

		syscall_installed_pids =
			segment.find_or_construct<syscall_pid_set>(
				bpftime::DEFAULT_SYSCALL_PID_SET_NAME)(
				std::less<int>(),
				syscall_pid_set_allocator(
					segment.get_segment_manager()));
		SPDLOG_DEBUG(
			"done: bpftime_shm for server setup: syscall_pid_set");

		agent_config = segment.find_or_construct<struct agent_config>(
			bpftime::DEFAULT_AGENT_CONFIG_NAME)();

		injected_pids = segment.find_or_construct<alive_agent_pids>(
			bpftime::DEFAULT_ALIVE_AGENT_PIDS_NAME)(
			std::less<int>(),
			alive_agent_pid_set_allocator(
				segment.get_segment_manager()));
		SPDLOG_DEBUG("done: bpftime_shm for open_or_create setup");
	} else if (type == shm_open_type::SHM_REMOVE_AND_CREATE) {
		SPDLOG_DEBUG(
			"start: bpftime_shm for server setup for memory size {}",
			memory_size);
		boost::interprocess::shared_memory_object::remove(shm_name);
		// create the shm
		SPDLOG_DEBUG(
			"done: bpftime_shm for server setup: remove installed segment");
		segment = boost::interprocess::managed_shared_memory(
			boost::interprocess::create_only,
			// Allocate 20M bytes of memory by default
			shm_name, memory_size << 20);
		SPDLOG_DEBUG("done: bpftime_shm for server setup: segment");

		manager = segment.construct<bpftime::handler_manager>(
			bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
		SPDLOG_DEBUG("done: bpftime_shm for server setup: manager");

		syscall_installed_pids = segment.construct<syscall_pid_set>(
			bpftime::DEFAULT_SYSCALL_PID_SET_NAME)(
			std::less<int>(),
			syscall_pid_set_allocator(
				segment.get_segment_manager()));
		SPDLOG_DEBUG(
			"done: bpftime_shm for server setup: syscall_pid_set");

		agent_config = segment.construct<struct agent_config>(
			bpftime::DEFAULT_AGENT_CONFIG_NAME)();
		SPDLOG_DEBUG(
			"done: bpftime_shm for server setup: agent_config");

		injected_pids = segment.construct<alive_agent_pids>(
			bpftime::DEFAULT_ALIVE_AGENT_PIDS_NAME)(
			std::less<int>(),
			alive_agent_pid_set_allocator(
				segment.get_segment_manager()));
		SPDLOG_DEBUG("done: bpftime_shm for server setup.");
	} else if (type == shm_open_type::SHM_NO_CREATE) {
		// not create any shm
		spdlog::warn(
			"NOT creating global shm. This is only for testing purpose.");
		return;
	}

#if BPFTIME_ENABLE_MPK
	// init mpk key
	pkey = pkey_alloc(0, PKEY_DISABLE_WRITE);
	if (pkey == -1) {
		SPDLOG_ERROR("pkey_alloc failed");
		return;
	}

	// protect shm segment
	if (pkey_mprotect(segment.get_address(), segment.get_size(),
			  PROT_READ | PROT_WRITE, pkey) == -1) {
		SPDLOG_ERROR("pkey_mprotect failed");
		return;
	}
	is_mpk_init = true;
#endif
}

bpftime_shm::bpftime_shm(bpftime::shm_open_type type)
	: bpftime_shm(bpftime::get_global_shm_name(), type)
{
	SPDLOG_INFO("Global shm constructed. shm_open_type {} for {}",
		    (int)type, bpftime::get_global_shm_name());
}

int bpftime_shm::add_bpf_map(int fd, const char *name,
			     bpftime::bpf_map_attr attr)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	if (!manager) {
		return -1;
	}
#ifdef ENABLE_BPFTIME_VERIFIER
	auto helpers = verifier::get_map_descriptors();
	helpers[fd] = verifier::BpftimeMapDescriptor{
		.original_fd = fd,
		.type = static_cast<uint32_t>(attr.type),
		.key_size = attr.key_size,
		.value_size = attr.value_size,
		.max_entries = attr.max_ents,
		.inner_map_fd = static_cast<unsigned int>(-1)
	};
	verifier::set_map_descriptors(helpers);
#endif
	return manager->set_handler(
		fd, bpftime::bpf_map_handler(fd, name, segment, attr), segment);
}

const handler_manager *bpftime_shm::get_manager() const
{
	return manager;
}

bool bpftime_shm::is_perf_event_handler_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	auto &handler = get_handler(fd);
	return std::holds_alternative<bpf_perf_event_handler>(handler);
}

bool bpftime_shm::is_software_perf_event_handler_fd(int fd) const
{
	if (!is_perf_event_handler_fd(fd))
		return false;
	const auto &hd = std::get<bpf_perf_event_handler>(get_handler(fd));
	return hd.type == (int)bpf_event_type::PERF_TYPE_SOFTWARE;
}

// local agent config can be used for test or local process
static agent_config local_agent_config = {};

void bpftime_shm::set_agent_config(const struct agent_config &config)
{
	if (agent_config == nullptr) {
		SPDLOG_INFO(
			"global agent_config is nullptr, set current process config");
		local_agent_config = config;
		return;
	}
	*agent_config = config;
}

const struct agent_config &bpftime_shm::get_agent_config()
{
	if (agent_config == nullptr) {
		SPDLOG_DEBUG("use current process config");
		return local_agent_config;
	}
	return *agent_config;
}

const bpftime::agent_config &bpftime_get_agent_config()
{
	return shm_holder.global_shared_memory.get_agent_config();
}

void bpftime_set_agent_config(const bpftime::agent_config &cfg)
{
	shm_holder.global_shared_memory.set_agent_config(cfg);
}

std::optional<void *>
bpftime_shm::get_software_perf_event_raw_buffer(int fd, size_t buffer_sz) const
{
	if (!is_software_perf_event_handler_fd(fd)) {
		SPDLOG_ERROR("Expected {} to be an perf event fd", fd);
		errno = EINVAL;
		return nullptr;
	}
	const auto &handler = std::get<bpf_perf_event_handler>(get_handler(fd));
	return handler.try_get_software_perf_data_raw_buffer(buffer_sz);
}
int bpftime_shm::add_custom_perf_event(int type, const char *attach_argument)
{
	int fd = open_fake_fd();
	if (fd < 0) {
		SPDLOG_ERROR("Unable to allocate id for custom perf event: {}",
			     errno);
		return fd;
	}
	manager->set_handler(
		fd, bpf_perf_event_handler(type, attach_argument, segment),
		segment);
	return fd;
}

void bpftime_shm::add_pid_into_alive_agent_set(int pid)
{
	injected_pids->insert(pid);
}
void bpftime_shm::remove_pid_from_alive_agent_set(int pid)
{
	injected_pids->erase(pid);
}
void bpftime_shm::iterate_all_pids_in_alive_agent_set(
	std::function<void(int)> &&cb)
{
	for (auto x : *injected_pids) {
		cb(x);
	}
}
} // namespace bpftime
