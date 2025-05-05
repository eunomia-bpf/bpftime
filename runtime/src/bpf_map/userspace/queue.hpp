/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, Your Name/Org Here (Adapt as needed)
 * All rights reserved.
 *
 * Implementation for BPF_MAP_TYPE_QUEUE mapping standard syscalls:
 * - BPF_MAP_UPDATE_ELEM -> Push (with flags)
 * - BPF_MAP_LOOKUP_ELEM -> Peek (returns internal pointer, requires caller copy)
 * - BPF_MAP_LOOKUP_AND_DELETE_ELEM -> Pop (delete part only, requires caller copy before delete)
 */
#ifndef BPFTIME_QUEUE_MAP_HPP
#define BPFTIME_QUEUE_MAP_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
// Include bpf.h for BPF flags like BPF_ANY, BPF_EXIST, etc.
#include <bpf/bpf.h>

// Forward declaration
namespace bpftime { class queue_map_impl; }

namespace bpftime
{

class queue_map_impl {
    private:
	// Internal mutex for thread/process safety
	boost::interprocess::interprocess_mutex mutex;

	// Circular buffer indices
	uint32_t head;
	uint32_t tail;

	// Configuration
	uint32_t _value_size;
	uint32_t capacity; // max_entries + 1
	uint32_t _max_entries;

	// Shared memory buffer pointer
	boost::interprocess::offset_ptr<uint8_t> buffer;

	// Allocator type and instance
	using char_allocator = boost::interprocess::allocator<
		uint8_t,
		boost::interprocess::managed_shared_memory::segment_manager>;
	char_allocator buffer_allocator;

	// Internal helpers
	bool is_full() const;
	bool is_empty() const;
	// Internal dequeue function used by push(BPF_EXIST) and delete
	// Must be called under lock.
	void internal_dequeue();
	// Internal enqueue function used by push
	// Must be called under lock.
	void internal_enqueue(const void *value);

    public:
	// Map handler should NOT lock externally, locking is internal.
	const static bool should_lock = false;

	/**
	 * @brief Construct a new queue map impl object.
	 * @param memory Managed shared memory segment.
	 * @param value_size Size of each element. Must be > 0.
	 * @param max_entries Maximum number of elements. Must be > 0.
	 */
	queue_map_impl(boost::interprocess::managed_shared_memory &memory,
		       uint32_t value_size, uint32_t max_entries);

	/**
	 * @brief Destroy the queue map impl object.
	 */
	~queue_map_impl();

	// Disable copy/assignment
	queue_map_impl(const queue_map_impl &) = delete;
	queue_map_impl &operator=(const queue_map_impl &) = delete;

	// --- Methods mapping to standard BPF syscall commands (within signature limits) ---

	/**
	 * @brief Implements BPF_MAP_LOOKUP_ELEM for Queue (Peek - Pointer Return).
	 * Returns a pointer to the element at the front of the queue without removing it.
	 * NOTE: Does NOT copy the value out. The caller (bpftime handler) is responsible
	 *       for copying the data from the returned pointer to the user's buffer.
	 * @param key Ignored (must be NULL conceptually for queue).
	 * @return Pointer to the element data in shared memory, or nullptr if the queue is empty.
	 *         The pointer is valid until the element is dequeued (via elem_delete).
	 */
	void *elem_lookup(const void *key);

	/**
	 * @brief Implements BPF_MAP_UPDATE_ELEM for Queue (Push operation with flags).
	 * Adds an element to the back of the queue according to flags.
	 * @param key Ignored (must be NULL conceptually for queue).
	 * @param value Pointer to the value data to enqueue. Must not be NULL.
	 * @param flags BPF_ANY (push only if not full) or BPF_EXIST (overwrite oldest if full).
	 *              Other flags result in -EINVAL.
	 * @return 0 on success. Negative error code on failure (-EBUSY if full and BPF_ANY, -EINVAL for bad flags/value).
	 */
	long elem_update(const void *key, const void *value, uint64_t flags);

	/**
	 * @brief Implements the delete part of BPF_MAP_LOOKUP_AND_DELETE_ELEM for Queue (Dequeue).
	 * Removes the element from the front of the queue. Does NOT return the element's value.
	 * NOTE: To implement the full Pop semantic, the caller (bpftime handler) must first
	 *       call elem_lookup, copy the data, and then call this elem_delete function.
	 *       Atomicity relies on the caller potentially re-acquiring the lock or
	 *       careful coordination if lock is external (which contradicts should_lock=false).
	 *       This function only performs the dequeue step.
	 * @param key Ignored (must be NULL conceptually for queue).
	 * @return 0 on success. Negative error code on failure (-ENOENT if empty).
	 */
	long elem_delete(const void *key);

	/**
	 * @brief Get next key. Not applicable for Queues.
	 * @param key Ignored.
	 * @param next_key Ignored.
	 * @return Always returns -EINVAL.
	 */
	int map_get_next_key(const void *key, void *next_key);

	// --- Helper methods ---
	uint32_t get_value_size() const;
	uint32_t get_max_entries() const;
	size_t get_current_size(); // Calculates current number of elements
};

} // namespace bpftime

#endif // BPFTIME_QUEUE_MAP_HPP