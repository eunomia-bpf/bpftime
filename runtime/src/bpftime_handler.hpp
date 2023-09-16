#ifndef _HANDLER_MANAGER_HPP
#define _HANDLER_MANAGER_HPP
#include <boost/interprocess/interprocess_fwd.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <variant>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/string.hpp>
#include "bpftime_shm.hpp"
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace bpftime
{
using managed_shared_memory = boost::interprocess::managed_shared_memory;
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;
using sharable_mutex_ptr = boost::interprocess::managed_unique_ptr<
	boost::interprocess::interprocess_sharable_mutex,
	boost::interprocess::managed_shared_memory>::type;
// bpf progs handler
// in share memory. This is only a simple data struct to store the
// bpf program data.
class bpf_prog_handler {
    public:
	/* Note that tracing related programs such as
	 * BPF_PROG_TYPE_{KPROBE,TRACEPOINT,PERF_EVENT,RAW_TRACEPOINT}
	 * are not subject to a stable API since kernel internal data
	 * structures can change from release to release and may
	 * therefore break existing tracing BPF programs. Tracing BPF
	 * programs correspond to /a/ specific kernel which is to be
	 * analyzed, and not /a/ specific kernel /and/ all future ones.
	 */
	enum class bpf_prog_type {
		BPF_PROG_TYPE_UNSPEC,
		BPF_PROG_TYPE_SOCKET_FILTER,
		BPF_PROG_TYPE_KPROBE,
		BPF_PROG_TYPE_SCHED_CLS,
		BPF_PROG_TYPE_SCHED_ACT,
		BPF_PROG_TYPE_TRACEPOINT,
		BPF_PROG_TYPE_XDP,
		BPF_PROG_TYPE_PERF_EVENT,
		BPF_PROG_TYPE_CGROUP_SKB,
		BPF_PROG_TYPE_CGROUP_SOCK,
		BPF_PROG_TYPE_LWT_IN,
		BPF_PROG_TYPE_LWT_OUT,
		BPF_PROG_TYPE_LWT_XMIT,
		BPF_PROG_TYPE_SOCK_OPS,
		BPF_PROG_TYPE_SK_SKB,
		BPF_PROG_TYPE_CGROUP_DEVICE,
		BPF_PROG_TYPE_SK_MSG,
		BPF_PROG_TYPE_RAW_TRACEPOINT,
		BPF_PROG_TYPE_CGROUP_SOCK_ADDR,
		BPF_PROG_TYPE_LWT_SEG6LOCAL,
		BPF_PROG_TYPE_LIRC_MODE2,
		BPF_PROG_TYPE_SK_REUSEPORT,
		BPF_PROG_TYPE_FLOW_DISSECTOR,
		BPF_PROG_TYPE_CGROUP_SYSCTL,
		BPF_PROG_TYPE_RAW_TRACEPOINT_WRITABLE,
		BPF_PROG_TYPE_CGROUP_SOCKOPT,
		BPF_PROG_TYPE_TRACING,
		BPF_PROG_TYPE_STRUCT_OPS,
		BPF_PROG_TYPE_EXT,
		BPF_PROG_TYPE_LSM,
		BPF_PROG_TYPE_SK_LOOKUP,
		BPF_PROG_TYPE_SYSCALL, /* a program that can execute syscalls */
		BPF_PROG_TYPE_NETFILTER,
	};
	enum bpf_prog_type type;

	bpf_prog_handler(managed_shared_memory &mem,
			 const struct ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
		: insns(shm_ebpf_inst_vector_allocator(
			  mem.get_segment_manager())),
		  attach_fds(
			  shm_int_vector_allocator(mem.get_segment_manager())),
		  name(char_allocator(mem.get_segment_manager()))
	{
		insns.assign(insn, insn + insn_cnt);
		this->name = prog_name;
	}
	bpf_prog_handler(const bpf_prog_handler &) = delete;
	bpf_prog_handler(bpf_prog_handler &&) noexcept = default;
	bpf_prog_handler &operator=(const bpf_prog_handler &) = delete;
	bpf_prog_handler &operator=(bpf_prog_handler &&) noexcept = default;

	typedef boost::interprocess::allocator<
		ebpf_inst, managed_shared_memory::segment_manager>
		shm_ebpf_inst_vector_allocator;

	typedef boost::interprocess::vector<ebpf_inst,
					    shm_ebpf_inst_vector_allocator>
		inst_vector;
	inst_vector insns;

	typedef boost::interprocess::allocator<
		int, managed_shared_memory::segment_manager>
		shm_int_vector_allocator;

	typedef boost::interprocess::vector<int, shm_int_vector_allocator>
		attach_fds_vector;
	mutable attach_fds_vector attach_fds;

	void add_attach_fd(int fd) const
	{
		attach_fds.push_back(fd);
	}

	boost_shm_string name;
};

// handle the bpf link fd
struct bpf_link_handler {
	uint32_t prog_fd, target_fd;
};

// perf event handler
struct bpf_perf_event_handler {
	enum class bpf_event_type {
		PERF_TYPE_HARDWARE = 0,
		PERF_TYPE_SOFTWARE = 1,
		PERF_TYPE_TRACEPOINT = 2,
		PERF_TYPE_HW_CACHE = 3,
		PERF_TYPE_RAW = 4,
		PERF_TYPE_BREAKPOINT = 5,

		// custom types
		BPF_TYPE_UPROBE = 6,
		BPF_TYPE_URETPROBE = 7,
		BPF_TYPE_FILTER = 8,
		BPF_TYPE_REPLACE = 9,
	} type;
	int enable() const
	{
		// TODO: implement enable logic.
		// If This is a server, should inject the agent into the target
		// process.
		return 0;
	}
	uint64_t offset;
	int pid;
	size_t ref_ctr_off;
	boost_shm_string _module_name;
	// Tracepoint id at /sys/kernel/tracing/events/syscalls/*/id, used to
	// indicate which syscall to trace
	int32_t tracepoint_id = -1;
	// attach to replace or filter self define types
	bpf_perf_event_handler(bpf_event_type type, uint64_t offset, int pid,
			       const char *module_name,
			       managed_shared_memory &mem)
		: type(type), offset(offset), pid(pid),
		  _module_name(char_allocator(mem.get_segment_manager()))
	{
		this->_module_name = module_name;
	}
	// create uprobe/uretprobe with new perf event attr
	bpf_perf_event_handler(bool is_retprobe, uint64_t offset, int pid,
			       const char *module_name, size_t ref_ctr_off,
			       managed_shared_memory &mem)
		: offset(offset), pid(pid), ref_ctr_off(ref_ctr_off),
		  _module_name(char_allocator(mem.get_segment_manager()))
	{
		if (is_retprobe) {
			type = bpf_event_type::BPF_TYPE_URETPROBE;
		} else {
			type = bpf_event_type::BPF_TYPE_UPROBE;
		}
		this->_module_name = module_name;
	}

	// create tracepoint
	bpf_perf_event_handler(int pid, int32_t tracepoint_id,
			       managed_shared_memory &mem)
		: pid(pid),
		  _module_name(char_allocator(mem.get_segment_manager())),
		  tracepoint_id(tracepoint_id)
	{
	}
};

// bpf map handler
// all map data will be put on shared memory, so it can be accessed by
// different processes
class bpf_map_handler {
    public:
	bpf_map_attr attr;
	using general_map_impl_ptr = boost::interprocess::offset_ptr<void>;
	enum bpf_map_type {
		BPF_MAP_TYPE_UNSPEC,
		BPF_MAP_TYPE_HASH,
		BPF_MAP_TYPE_ARRAY,
		BPF_MAP_TYPE_PROG_ARRAY,
		BPF_MAP_TYPE_PERF_EVENT_ARRAY,
		BPF_MAP_TYPE_PERCPU_HASH,
		BPF_MAP_TYPE_PERCPU_ARRAY,
		BPF_MAP_TYPE_STACK_TRACE,
		BPF_MAP_TYPE_CGROUP_ARRAY,
		BPF_MAP_TYPE_LRU_HASH,
		BPF_MAP_TYPE_LRU_PERCPU_HASH,
		BPF_MAP_TYPE_LPM_TRIE,
		BPF_MAP_TYPE_ARRAY_OF_MAPS,
		BPF_MAP_TYPE_HASH_OF_MAPS,
		BPF_MAP_TYPE_DEVMAP,
		BPF_MAP_TYPE_SOCKMAP,
		BPF_MAP_TYPE_CPUMAP,
		BPF_MAP_TYPE_XSKMAP,
		BPF_MAP_TYPE_SOCKHASH,
		BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED,
		/* BPF_MAP_TYPE_CGROUP_STORAGE is available to bpf programs
		 * attaching to a cgroup. The newer BPF_MAP_TYPE_CGRP_STORAGE is
		 * available to both cgroup-attached and other progs and
		 * supports all functionality provided by
		 * BPF_MAP_TYPE_CGROUP_STORAGE. So mark
		 * BPF_MAP_TYPE_CGROUP_STORAGE deprecated.
		 */
		BPF_MAP_TYPE_CGROUP_STORAGE =
			BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED,
		BPF_MAP_TYPE_REUSEPORT_SOCKARRAY,
		BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE,
		BPF_MAP_TYPE_QUEUE,
		BPF_MAP_TYPE_STACK,
		BPF_MAP_TYPE_SK_STORAGE,
		BPF_MAP_TYPE_DEVMAP_HASH,
		BPF_MAP_TYPE_STRUCT_OPS,
		BPF_MAP_TYPE_RINGBUF,
		BPF_MAP_TYPE_INODE_STORAGE,
		BPF_MAP_TYPE_TASK_STORAGE,
		BPF_MAP_TYPE_BLOOM_FILTER,
		BPF_MAP_TYPE_USER_RINGBUF,
		BPF_MAP_TYPE_CGRP_STORAGE,
	};
	enum bpf_map_type type;
	boost_shm_string name;
	bpf_map_handler(const char *name, managed_shared_memory &mem,
			bpf_map_attr attr)
		: bpf_map_handler(attr.type, attr.key_size, attr.value_size,
				  attr.max_ents, attr.flags, name, mem)
	{
		this->attr = attr;
	}
	bpf_map_handler(int type, uint32_t key_size, uint32_t value_size,
			uint32_t max_ents, uint64_t flags, const char *name,
			managed_shared_memory &mem)
		: type((bpf_map_type)type),
		  name(char_allocator(mem.get_segment_manager())),
		  map_mutex(boost::interprocess::make_managed_unique_ptr(
			  mem.construct<boost::interprocess::
						interprocess_sharable_mutex>(
				  boost::interprocess::anonymous_instance)(),
			  mem)),
		  map_impl_ptr(nullptr), max_entries(max_ents), flags(flags),
		  key_size(key_size), value_size(value_size)

	{
		this->name = name;
	}
	bpf_map_handler(const bpf_map_handler &) = delete;
	bpf_map_handler(bpf_map_handler &&) noexcept = default;
	bpf_map_handler &operator=(const bpf_map_handler &) = delete;
	bpf_map_handler &operator=(bpf_map_handler &&) noexcept = default;
	~bpf_map_handler()
	{
		// since we cannot free here because memory allocator pointer
		// cannot be held between process, we will free the internal map
		// in the handler_manager.
		assert(map_impl_ptr.get() == nullptr);
	}
	//  * BPF_MAP_LOOKUP_ELEM
	// *	Description
	// *		Look up an element with a given *key* in the map
	// * 		referred to by the file descriptor *map_fd*.
	// *
	// *		The *flags* argument may be specified as one of the
	// *		following:
	// *
	// *		**BPF_F_LOCK**
	// *			Look up the value of a spin-locked map without
	// *			returning the lock. This must be specified if
	// * 			the elements contain a spinlock.
	// *
	// *	Return
	// *		Returns zero on success. On error, -1 is returned and
	// *  		*errno* is set appropriately.
	// *
	const void *map_lookup_elem(const void *key) const;
	long map_update_elem(const void *key, const void *value,
			     uint64_t flags) const;
	long map_delete_elem(const void *key) const;
	// * BPF_MAP_GET_NEXT_KEY
	// *	Description
	// *		Look up an element by key in a specified map and return
	// *		the key of the next element. Can be used to iterate over
	// *		all elements in the map.
	// *
	// *	Return
	// *		Returns zero on success. On error, -1 is returned and
	// *		*errno* is set appropriately.
	// *
	// *		The following cases can be used to iterate over all
	// *		elements of the map:
	// *
	// *		* If *key* is not found, the operation returns zero and
	// *		sets the *next_key* pointer to the key of the first
	// *		element.  If *key* is found, the operation returns zero
	// *		and sets the *next_key* pointer to the key of the
	// *		next element. * If *key* is the last element, returns
	// *		-1 and *errno* is set to **ENOENT**.
	// *
	// *		May set *errno* to **ENOMEM**, **EFAULT**, **EPERM**, or
	// *		**EINVAL** on error.
	// *
	int bpf_map_get_next_key(const void *key, void *next_key) const;
	void map_free(managed_shared_memory &memory);
	int map_init(managed_shared_memory &memory);
	uint32_t get_value_size() const
	{
		return value_size;
	}

    private:
	std::string get_container_name()
	{
		return "ebpf_map_fd_" + std::string(name.c_str());
	}
	mutable sharable_mutex_ptr map_mutex;
	// The underlying data structure of the map
	general_map_impl_ptr map_impl_ptr;
	uint32_t max_entries = 0;
	uint64_t flags = 0;
	uint32_t key_size = 0;
	uint32_t value_size = 0;
};

const size_t DEFAULT_MAX_FD = 1024;

struct unused_handler {};

using boost::interprocess::allocator;
using boost::interprocess::vector;

constexpr const char *DEFAULT_GLOBAL_SHM_NAME = "bpftime_maps_shm";
constexpr const char *DEFAULT_GLOBAL_HANDLER_NAME = "bpftime_handler";

inline const char *get_global_shm_name()
{
	const char *name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	if (name == nullptr) {
		return DEFAULT_GLOBAL_SHM_NAME;
	}
	return name;
}

struct shm_remove {
	shm_remove()
	{
		boost::interprocess::shared_memory_object::remove(
			get_global_shm_name());
	}
	shm_remove(const char *name)
	{
		boost::interprocess::shared_memory_object::remove(name);
	}
	~shm_remove()
	{
		std::cout << "remove shm " << get_global_shm_name()
			  << std::endl;
		boost::interprocess::shared_memory_object::remove(
			get_global_shm_name());
	}
};

// handler manager for keep bpf maps and progs fds
// This struct will be put on shared memory
class handler_manager {
    public:
	using handler_variant =
		std::variant<unused_handler, bpf_map_handler, bpf_link_handler,
			     bpf_prog_handler, bpf_perf_event_handler>;

	using handler_variant_allocator =
		allocator<handler_variant,
			  managed_shared_memory::segment_manager>;

	using handler_variant_vector =
		boost::interprocess::vector<handler_variant,
					    handler_variant_allocator>;

	handler_manager(managed_shared_memory &mem,
			size_t max_fd_count = DEFAULT_MAX_FD)
		: handlers(max_fd_count,
			   handler_variant_allocator(mem.get_segment_manager()))
	{
	}

	~handler_manager()
	{
		for (std::size_t i = 0; i < handlers.size(); i++) {
			assert(!is_allocated(i));
		}
	}

	const handler_variant &get_handler(int fd) const
	{
		return handlers[fd];
	}

	const handler_variant &operator[](int idx) const
	{
		return handlers[idx];
	}
	std::size_t size() const
	{
		return handlers.size();
	}

	void set_handler(int fd, handler_variant &&handler,
			 managed_shared_memory &memory)
	{
		handlers[fd] = std::move(handler);
		if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
			std::get<bpf_map_handler>(handlers[fd]).map_init(memory);
		}
	}

	bool is_allocated(int fd) const
	{
		if (fd < 0 || (std::size_t)fd >= handlers.size()) {
			return false;
		}
		return !std::holds_alternative<unused_handler>(handlers.at(fd));
	}

	void clear_fd_at(int fd, managed_shared_memory &memory)
	{
		if (fd < 0 || (std::size_t)fd >= handlers.size()) {
			return;
		}
		if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
			std::get<bpf_map_handler>(handlers[fd]).map_free(memory);
		}
		handlers[fd] = unused_handler();
	}

	void clear_all(managed_shared_memory &memory)
	{
		for (std::size_t i = 0; i < handlers.size(); i++) {
			if (is_allocated(i)) {
				clear_fd_at(i, memory);
			}
		}
	}

	handler_manager(const handler_manager &) = delete;
	handler_manager(handler_manager &&) noexcept = default;
	handler_manager &operator=(const handler_manager &) = delete;
	handler_manager &operator=(handler_manager &&) noexcept = default;

    private:
	handler_variant_vector handlers;
};

// global bpftime share memory
class bpftime_shm {
	// shared memory segment
	boost::interprocess::managed_shared_memory segment;

	// manage the bpf fds in the shared memory
	bpftime::handler_manager *manager = nullptr;

    public:
	const handler_manager::handler_variant &get_handler(int fd) const
	{
		return manager->get_handler(fd);
	}

	bool is_map_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_map_handler>(
			handler);
	}

	bool is_prog_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_prog_handler>(
			handler);
	}

	bool is_perf_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		const auto &handler = manager->get_handler(fd);
		return std::holds_alternative<bpftime::bpf_perf_event_handler>(
			handler);
	}

	int open_fake_fd()
	{
		return open("/dev/null", O_RDONLY);
	}

	// handle bpf commands to load a bpf program
	int add_bpf_prog(const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type)
	{
		int fd = open_fake_fd();
		manager->set_handler(
			fd,
			bpftime::bpf_prog_handler(segment, insn, insn_cnt,
						  prog_name, prog_type),
			segment);
		return fd;
	}

	// add a bpf link fd
	int add_bpf_link(int prog_fd, int target_fd)
	{
		int fd = open_fake_fd();
		if (!manager->is_allocated(target_fd) || !is_prog_fd(prog_fd)) {
			return -1;
		}
		manager->set_handler(
			fd,
			bpftime::bpf_link_handler{ (uint32_t)prog_fd,
						   (uint32_t)target_fd },
			segment);
		return fd;
	}

	// create a bpf map fd
	int add_bpf_map(const char *name, bpftime::bpf_map_attr attr)
	{
		int fd = open_fake_fd();
		if (!manager) {
			return -1;
		}
		manager->set_handler(
			fd, bpftime::bpf_map_handler(name, segment, attr),
			segment);
		return fd;
	}
	uint32_t bpf_map_value_size(int fd) const;
	const void *bpf_map_lookup_elem(int fd, const void *key) const;

	long bpf_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags) const;

	long bpf_delete_elem(int fd, const void *key) const;

	int bpf_map_get_next_key(int fd, const void *key, void *next_key) const;

	// create an uprobe fd
	int add_uprobe(int pid, const char *name, uint64_t offset,
		       bool retprobe, size_t ref_ctr_off);
	int add_tracepoint(int pid, int32_t tracepoint_id);
	int attach_perf_to_bpf(int perf_fd, int bpf_fd);
	int attach_enable(int fd) const;
	// remove a fake fd from the manager.
	// The fake fd should be closed by the caller.
	void close_fd(int fd)
	{
		if (manager) {
			manager->clear_fd_at(fd, segment);
		}
	}

	bool is_exist_fake_fd(int fd) const
	{
		if (manager == nullptr || fd < 0 ||
		    (std::size_t)fd >= manager->size()) {
			return false;
		}
		return manager->is_allocated(fd);
	}

	bpftime_shm()
	{
		std::cout << "global_shm_open_type "
			  << (int)global_shm_open_type << " for "
			  << bpftime::get_global_shm_name() << std::endl;
		if (global_shm_open_type == shm_open_type::SHM_CLIENT) {
			// open the shm
			segment = boost::interprocess::managed_shared_memory(
				boost::interprocess::open_only,
				bpftime::get_global_shm_name());
			manager =
				segment.find<bpftime::handler_manager>(
					       bpftime::DEFAULT_GLOBAL_HANDLER_NAME)
					.first;
		} else if (global_shm_open_type == shm_open_type::SHM_SERVER) {
			boost::interprocess::shared_memory_object::remove(
				bpftime::get_global_shm_name());
			// create the shm
			segment = boost::interprocess::managed_shared_memory(
				boost::interprocess::create_only,
				bpftime::get_global_shm_name(), 1 << 20);
			manager = segment.construct<bpftime::handler_manager>(
				bpftime::DEFAULT_GLOBAL_HANDLER_NAME)(segment);
		} else if (global_shm_open_type ==
			   shm_open_type::SHM_NO_CREATE) {
			// not create any shm
			return;
		}
	}

	const handler_manager *get_manager() const
	{
		return manager;
	}
};

} // namespace bpftime

#endif
