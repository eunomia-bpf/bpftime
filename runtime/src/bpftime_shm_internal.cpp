#include "bpftime_shm.hpp"
#include "handler/epoll_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_shm_internal.hpp>
#include <cstdio>
#include <sys/epoll.h>
#include <unistd.h>
#include <variant>

static bool global_shm_initialized = false;

extern "C" void bpftime_initialize_global_shm(bpftime::shm_open_type type)
{
	using namespace bpftime;
	// Use placement new, which will not allocate memory, but just
	// call the constructor
	new (&shm_holder.global_shared_memory) bpftime_shm(type);
	global_shm_initialized = true;
}

extern "C" void bpftime_destroy_global_shm()
{
	using namespace bpftime;
	if (global_shm_initialized) {
		// spdlog::info("Global shm destructed");
		shm_holder.global_shared_memory.~bpftime_shm();
		// Why not spdlog? because global variables that spdlog used
		// were already destroyed..
		printf("INFO [%d]: Global shm destructed\n", (int)getpid());
	}
}

extern "C" void bpftime_remove_global_shm()
{
	using namespace bpftime;
	boost::interprocess::shared_memory_object::remove(
		get_global_shm_name());
	spdlog::info("Global shm removed");
}

static __attribute__((destructor(65535))) void __destruct_shm()
{
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
					     bool from_userspace) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return nullptr;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_lookup_elem(key, from_userspace);
}

long bpftime_shm::bpf_map_update_elem(int fd, const void *key,
				      const void *value, uint64_t flags,
				      bool from_userspace) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_update_elem(key, value, flags, from_userspace);
}

long bpftime_shm::bpf_delete_elem(int fd, const void *key,
				  bool from_userspace) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.map_delete_elem(key, from_userspace);
}

int bpftime_shm::bpf_map_get_next_key(int fd, const void *key, void *next_key,
				      bool from_userspace) const
{
	if (!is_map_fd(fd)) {
		errno = ENOENT;
		return -1;
	}
	auto &handler =
		std::get<bpftime::bpf_map_handler>(manager->get_handler(fd));
	return handler.bpf_map_get_next_key(key, next_key, from_userspace);
}

int bpftime_shm::add_uprobe(int fd, int pid, const char *name, uint64_t offset,
			    bool retprobe, size_t ref_ctr_off)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	spdlog::debug("Set fd {} to uprobe, pid={}, name={}, offset={}", fd,
		      pid, name, offset);
	return manager->set_handler(
		fd,
		bpftime::bpf_perf_event_handler{ retprobe, offset, pid, name,
						 ref_ctr_off, segment },
		segment);
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
	int fd = open_fake_fd();
	return manager->set_handler(fd,
				    bpftime::bpf_perf_event_handler(
					    cpu, sample_type, config, segment),
				    segment);
	return fd;
}

int bpftime_shm::attach_perf_to_bpf(int perf_fd, int bpf_fd)
{
	if (!is_perf_fd(perf_fd)) {
		spdlog::error("Fd {} is not a perf fd", perf_fd);
		errno = ENOENT;
		return -1;
	}
	return add_bpf_prog_attach_target(perf_fd, bpf_fd);
}

int bpftime_shm::add_bpf_prog_attach_target(int perf_fd, int bpf_fd)
{
	spdlog::debug("Try attaching prog fd {} to perf fd {}", bpf_fd,
		      perf_fd);
	if (!is_prog_fd(bpf_fd)) {
		spdlog::error("Fd {} is not a prog fd", bpf_fd);
		errno = ENOENT;
		return -1;
	}
	auto &handler = std::get<bpftime::bpf_prog_handler>(
		manager->get_handler(bpf_fd));
	handler.add_attach_fd(perf_fd);
	return 0;
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
		spdlog::error("Fd {} is expected to be an epoll fd", epoll_fd);
		errno = EINVAL;
		return -1;
	}
	auto &epoll_inst =
		std::get<epoll_handler>(manager->get_handler(epoll_fd));
	if (!is_software_perf_event_handler_fd(swpe_fd)) {
		spdlog::error(
			"Fd {} is expected to be an software perf event handler",
			swpe_fd);
		errno = EINVAL;
		return -1;
	}
	auto &perf_handler =
		std::get<bpf_perf_event_handler>(manager->get_handler(swpe_fd));
	if (perf_handler.type != bpf_event_type::PERF_TYPE_SOFTWARE) {
		spdlog::error(
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
		spdlog::error(
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
		epoll_inst.files.emplace_back(val->create_impl_weak_ptr(),
					      extra_data);
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
	fd = manager->set_handler(fd, bpftime::epoll_handler(segment), segment);
	spdlog::debug("Epoll instance created: fd={}", fd);
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
		spdlog::error("Invalid fd: {}", fd);
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
std::optional<ringbuf_map_impl *>
bpftime_shm::try_get_ringbuf_map_impl(int fd) const
{
	if (!is_ringbuf_map_fd(fd)) {
		spdlog::error("Expected fd {} to be an ringbuf map fd", fd);
		return {};
	}
	auto &map_handler = std::get<bpf_map_handler>(manager->get_handler(fd));
	return map_handler.try_get_ringbuf_map_impl();
}

std::optional<array_map_impl *>
bpftime_shm::try_get_array_map_impl(int fd) const
{
	if (!is_array_map_fd(fd)) {
		spdlog::error("Expected fd {} to be an array map fd", fd);
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
	return open("/dev/null", O_RDONLY);
}

// handle bpf commands to load a bpf program
int bpftime_shm::add_bpf_prog(int fd, const ebpf_inst *insn, size_t insn_cnt,
			      const char *prog_name, int prog_type)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	spdlog::debug(
		"Set handler fd {} to bpf_prog_handler, name {}, prog_type {}, insn_cnt {}",
		fd, prog_name, prog_type, insn_cnt);
	return manager->set_handler(
		fd,
		bpftime::bpf_prog_handler(segment, insn, insn_cnt, prog_name,
					  prog_type),
		segment);
}

// add a bpf link fd
int bpftime_shm::add_bpf_link(int fd, int prog_fd, int target_fd)
{
	if (fd < 0) {
		// if fd is negative, we need to create a new fd for allocating
		fd = open_fake_fd();
	}
	if (!manager->is_allocated(target_fd) || !is_prog_fd(prog_fd)) {
		return -1;
	}
	return manager->set_handler(
		fd,
		bpftime::bpf_link_handler{ (uint32_t)prog_fd,
					   (uint32_t)target_fd },
		segment);
}

void bpftime_shm::close_fd(int fd)
{
	if (manager) {
		manager->clear_fd_at(fd, segment);
	}
}

bool bpftime_shm::is_exist_fake_fd(int fd) const
{
	if (manager == nullptr || fd < 0 ||
	    (std::size_t)fd >= manager->size()) {
		return false;
	}
	return manager->is_allocated(fd);
}

bpftime_shm::bpftime_shm(const char *shm_name, shm_open_type type)
{
	if (type == shm_open_type::SHM_OPEN_ONLY) {
		spdlog::debug("start: bpftime_shm for client setup");
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
		spdlog::debug("done: bpftime_shm for client setup");
	} else if (type == shm_open_type::SHM_CREATE_OR_OPEN) {
		spdlog::debug("start: bpftime_shm for create or open setup");
		segment = boost::interprocess::managed_shared_memory(
			boost::interprocess::open_or_create,
			// Allocate 20M bytes of memory by default
			shm_name, 20 << 20);

		manager = segment.find_or_construct<bpftime::handler_manager>(
			bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
		spdlog::debug("done: bpftime_shm for server setup: manager");

		syscall_installed_pids =
			segment.find_or_construct<syscall_pid_set>(
				bpftime::DEFAULT_SYSCALL_PID_SET_NAME)(
				std::less<int>(),
				syscall_pid_set_allocator(
					segment.get_segment_manager()));
		spdlog::debug(
			"done: bpftime_shm for server setup: syscall_pid_set");

		agent_config = segment.find_or_construct<struct agent_config>(
			bpftime::DEFAULT_AGENT_CONFIG_NAME)();
		spdlog::debug("done: bpftime_shm for open_or_create setup");
	} else if (type == shm_open_type::SHM_REMOVE_AND_CREATE) {
		spdlog::debug("start: bpftime_shm for server setup");
		boost::interprocess::shared_memory_object::remove(shm_name);
		// create the shm
		spdlog::debug(
			"done: bpftime_shm for server setup: remove installed segment");
		segment = boost::interprocess::managed_shared_memory(
			boost::interprocess::create_only,
			// Allocate 20M bytes of memory by default
			shm_name, 20 << 20);
		spdlog::debug("done: bpftime_shm for server setup: segment");

		manager = segment.construct<bpftime::handler_manager>(
			bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
		spdlog::debug("done: bpftime_shm for server setup: manager");

		syscall_installed_pids = segment.construct<syscall_pid_set>(
			bpftime::DEFAULT_SYSCALL_PID_SET_NAME)(
			std::less<int>(),
			syscall_pid_set_allocator(
				segment.get_segment_manager()));
		spdlog::debug(
			"done: bpftime_shm for server setup: syscall_pid_set");

		agent_config = segment.construct<struct agent_config>(
			bpftime::DEFAULT_AGENT_CONFIG_NAME)();
		spdlog::debug(
			"done: bpftime_shm for server setup: agent_config");
		spdlog::debug("done: bpftime_shm for server setup.");
	} else if (type == shm_open_type::SHM_NO_CREATE) {
		// not create any shm
		spdlog::warn(
			"NOT creating global shm. This is only for testing purpose.");
		return;
	}
}

bpftime_shm::bpftime_shm(bpftime::shm_open_type type)
	: bpftime_shm(bpftime::get_global_shm_name(), type)
{
	spdlog::info("Global shm constructed. shm_open_type {} for {}",
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
		fd, bpftime::bpf_map_handler(name, segment, attr), segment);
}

const handler_manager *bpftime_shm::get_manager() const
{
	return manager;
}

bool bpftime_shm::is_perf_event_handler_fd(int fd) const
{
	auto &handler = get_handler(fd);
	return std::holds_alternative<bpf_perf_event_handler>(handler);
}

bool bpftime_shm::is_software_perf_event_handler_fd(int fd) const
{
	if (!is_perf_event_handler_fd(fd))
		return false;
	const auto &hd = std::get<bpf_perf_event_handler>(get_handler(fd));
	return hd.type == bpf_event_type::PERF_TYPE_SOFTWARE;
}

void bpftime_shm::set_agent_config(const struct agent_config &config)
{
	if (agent_config == nullptr) {
		spdlog::error("agent_config is nullptr, set error");
		return;
	}
	*agent_config = config;
}

const bpftime::agent_config &bpftime_get_agent_config()
{
	return shm_holder.global_shared_memory.get_agent_config();
}

void bpftime_set_agent_config(bpftime::agent_config &cfg)
{
	shm_holder.global_shared_memory.set_agent_config(cfg);
}

std::optional<void *>
bpftime_shm::get_software_perf_event_raw_buffer(int fd, size_t buffer_sz) const
{
	if (!is_software_perf_event_handler_fd(fd)) {
		spdlog::error("Expected {} to be an perf event fd", fd);
		errno = EINVAL;
		return nullptr;
	}
	const auto &handler = std::get<bpf_perf_event_handler>(get_handler(fd));
	return handler.try_get_software_perf_data_raw_buffer(buffer_sz);
}

} // namespace bpftime
