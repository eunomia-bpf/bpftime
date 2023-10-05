#ifndef _MAP_HANDLER
#define _MAP_HANDLER
#include "bpftime_shm.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
namespace bpftime
{
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;

using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

using sharable_mutex_ptr = boost::interprocess::managed_unique_ptr<
	boost::interprocess::interprocess_sharable_mutex,
	boost::interprocess::managed_shared_memory>::type;
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
	bpf_map_handler(const char *name,
			boost::interprocess::managed_shared_memory &mem,
			bpf_map_attr attr)
		: bpf_map_handler(attr.type, attr.key_size, attr.value_size,
				  attr.max_ents, attr.flags, name, mem)
	{
		this->attr = attr;
	}
	bpf_map_handler(int type, uint32_t key_size, uint32_t value_size,
			uint32_t max_ents, uint64_t flags, const char *name,
			boost::interprocess::managed_shared_memory &mem)
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
	void map_free(boost::interprocess::managed_shared_memory &memory);
	int map_init(boost::interprocess::managed_shared_memory &memory);
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
	[[maybe_unused]] uint64_t flags = 0;
	uint32_t key_size = 0;
	uint32_t value_size = 0;
};

} // namespace bpftime
#endif
