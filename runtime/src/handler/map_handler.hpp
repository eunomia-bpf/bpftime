/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_HANDLER
#define _MAP_HANDLER
#include "bpf_map/userspace/array_map.hpp"
#include "bpf_map/userspace/ringbuf_map.hpp"
#include "bpftime_shm.hpp"
#include "spdlog/spdlog.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <optional>
#include <unistd.h>
#include <bpf_map/shared/perf_event_array_kernel_user.hpp>
#if __APPLE__
#include "spinlock_wrapper.hpp"
#endif
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

// lock guad for RAII
class bpftime_lock_guard {
    private:
	volatile pthread_spinlock_t &spinlock;

    public:
	explicit bpftime_lock_guard(volatile pthread_spinlock_t &spinlock)
		: spinlock(spinlock)
	{
		pthread_spin_lock(&spinlock);
	}
	~bpftime_lock_guard()
	{
		pthread_spin_unlock(&spinlock);
	}
	// Delete copy constructor and assignment operator
	bpftime_lock_guard(const bpftime_lock_guard &) = delete;
	bpftime_lock_guard &operator=(const bpftime_lock_guard &) = delete;
};

// bpf map handler
// all map data will be put on shared memory, so it can be accessed by
// different processes
class bpf_map_handler {
    public:
	bpf_map_attr attr;
	using general_map_impl_ptr = boost::interprocess::offset_ptr<void>;
	bpf_map_type type;
	boost_shm_string name;
	bpf_map_handler(int id, const char *name,
			boost::interprocess::managed_shared_memory &mem,
			bpf_map_attr attr)
		: bpf_map_handler(id, attr.type, attr.key_size, attr.value_size,
				  attr.max_ents, attr.flags, name, mem)
	{
		this->attr = attr;
	}
	bpf_map_handler(int id, int type, uint32_t key_size,
			uint32_t value_size, uint32_t max_ents, uint64_t flags,
			const char *name,
			boost::interprocess::managed_shared_memory &mem)
		: type((bpf_map_type)type),
		  name(char_allocator(mem.get_segment_manager())),
		  map_impl_ptr(nullptr), max_entries(max_ents), flags(flags),
		  key_size(key_size), value_size(value_size)

	{
		SPDLOG_DEBUG("Create map with type {}", type);
		pthread_spin_init(&map_lock, 0);
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
		if (map_impl_ptr.get() != nullptr) {
			SPDLOG_CRITICAL(
				"Map impl of id {} is not freed when the map was being destroyed. This should not happen",
				id);
		}
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
	const void *map_lookup_elem(const void *key,
				    bool from_syscall = false) const;
	// * BPF_MAP_UPDATE_ELEM
	// *	Description
	// *		Create or update an element (key/value pair) in a
	// specified map.
	// *
	// *		The *flags* argument should be specified as one of the
	// *		following:
	// *
	// *		**BPF_ANY**
	// *			Create a new element or update an existing
	// element. *		**BPF_NOEXIST** *			Create a
	// new element only if it did not exist. *		**BPF_EXIST** *
	// Update an existing element. *		**BPF_F_LOCK** *
	// Update a spin_lock-ed map element.
	// *
	// *	Return
	// *		Returns zero on success. On error, -1 is returned and
	// *errno* *		is set appropriately.
	// *
	// *		May set *errno* to **EINVAL**, **EPERM**, **ENOMEM**,
	// *		**E2BIG**, **EEXIST**, or **ENOENT**.
	// *
	// *		**E2BIG**
	// *			The number of elements in the map reached the
	// *			*max_entries* limit specified at map creation
	// time. *		**EEXIST** *			If *flags*
	// specifies **BPF_NOEXIST** and the element *			with
	// *key* already exists in the map. *		**ENOENT** *
	// If *flags* specifies **BPF_EXIST** and the element with *
	// *key* does not exist in the map.
	// *
	long map_update_elem(const void *key, const void *value, uint64_t flags,
			     bool from_syscall = false) const;
	// * BPF_MAP_DELETE_ELEM
	// *	Description
	// *		Look up and delete an element by key in a specified map.
	// *
	// *	Return
	// *		Returns zero on success. On error, -1 is returned and
	// *errno* *		is set appropriately.
	long map_delete_elem(const void *key, bool from_syscall = false) const;
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
	int bpf_map_get_next_key(const void *key, void *next_key,
				 bool from_syscall = false) const;
	void map_free(boost::interprocess::managed_shared_memory &memory);
	int map_init(boost::interprocess::managed_shared_memory &memory);
	uint32_t get_value_size() const;
	std::optional<ringbuf_map_impl *> try_get_ringbuf_map_impl() const;
	std::optional<array_map_impl *> try_get_array_map_impl() const;
	std::optional<perf_event_array_kernel_user_impl *>
	try_get_shared_perf_event_array_map_impl() const;

    private:
	int id = 0;
	std::string get_container_name();
	mutable pthread_spinlock_t map_lock;
	// The underlying data structure of the map
	general_map_impl_ptr map_impl_ptr;
	uint32_t max_entries = 0;
	[[maybe_unused]] uint64_t flags = 0;
	uint32_t key_size = 0;
	uint32_t value_size = 0;
};

} // namespace bpftime
#endif
