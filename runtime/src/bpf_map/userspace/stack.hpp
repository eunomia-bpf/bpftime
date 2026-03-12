/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, Your Name/Org Here (Adapt as needed)
 * All rights reserved.
 *
 * Implementation for BPF_MAP_TYPE_STACK mapping both standard syscalls and new
 * helper functions:
 * - BPF_MAP_UPDATE_ELEM -> Push (with flags)
 * - BPF_MAP_LOOKUP_ELEM -> Peek (returns internal pointer, requires caller
 * copy)
 * - BPF_MAP_LOOKUP_AND_DELETE_ELEM -> Pop (delete part only, requires caller
 * copy before delete)
 * - bpf_map_push_elem -> Push (with flags)
 * - bpf_map_pop_elem -> Pop (copies value and removes)
 * - bpf_map_peek_elem -> Peek (copies value without removing)
 *
 * STACK implements LIFO (Last In, First Out) behavior:
 * - Push adds elements to the top of the stack
 * - Pop/Peek operates on the top of the stack (most recently added element)
 */
#ifndef BPFTIME_STACK_MAP_HPP
#define BPFTIME_STACK_MAP_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
// Include bpf flags for BPF_ANY, BPF_EXIST, etc.
#ifdef BPFTIME_BUILD_WITH_LIBBPF
#include <bpf/bpf.h>
#else
#include "bpf_flags.hpp"
#endif

// Forward declaration
namespace bpftime
{
class stack_map_impl;
}

namespace bpftime
{

class stack_map_impl {
    private:
	// Configuration
	unsigned int _value_size;
	unsigned int _max_entries;

	// Vector type and allocator for storing stack elements
	using byte_allocator = boost::interprocess::allocator<
		uint8_t,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using byte_vector =
		boost::interprocess::vector<uint8_t, byte_allocator>;

	// Vector to store stack data
	byte_vector data;

	// Internal helpers
	bool is_full() const;
	bool is_empty() const;

    public:
	// Map handler should lock externally
	const static bool should_lock = true;

	/**
	 * @brief Construct a new stack map impl object.
	 * @param memory Managed shared memory segment.
	 * @param value_size Size of each element. Must be > 0.
	 * @param max_entries Maximum number of elements. Must be > 0.
	 */
	stack_map_impl(boost::interprocess::managed_shared_memory &memory,
		       unsigned int value_size, unsigned int max_entries);

	/**
	 * @brief Destroy the stack map impl object.
	 */
	~stack_map_impl() = default;

	// Disable copy/assignment
	stack_map_impl(const stack_map_impl &) = delete;
	stack_map_impl &operator=(const stack_map_impl &) = delete;

	// --- Methods mapping to standard BPF syscall commands (within
	// signature limits) ---

	/**
	 * @brief Implements BPF_MAP_LOOKUP_ELEM for Stack (Peek - Pointer
	 * Return). Returns a pointer to the element at the top of the stack
	 * without removing it. NOTE: Does NOT copy the value out. The caller
	 * (bpftime handler) is responsible for copying the data from the
	 * returned pointer to the user's buffer.
	 * @param key Ignored (must be NULL conceptually for stack).
	 * @return Pointer to the element data in shared memory, or nullptr if
	 * the stack is empty. The pointer is valid until the element is
	 * popped (via elem_delete).
	 */
	void *elem_lookup(const void *key);

	/**
	 * @brief Implements BPF_MAP_UPDATE_ELEM for Stack (Push operation with
	 * flags). Adds an element to the top of the stack according to flags.
	 * @param key Ignored (must be NULL conceptually for stack).
	 * @param value Pointer to the value data to push. Must not be NULL.
	 * @param flags BPF_ANY (push only if not full) or BPF_EXIST (overwrite
	 * oldest if full). Other flags result in -EINVAL.
	 * @return 0 on success. Negative error code on failure (-EBUSY if full
	 * and BPF_ANY, -EINVAL for bad flags/value).
	 */
	long elem_update(const void *key, const void *value, uint64_t flags);

	/**
	 * @brief Implements the delete part of BPF_MAP_LOOKUP_AND_DELETE_ELEM
	 * for Stack (Pop). Removes the element from the top of the stack.
	 * Does NOT return the element's value. NOTE: To implement the full Pop
	 * semantic, the caller (bpftime handler) must first call elem_lookup,
	 * copy the data, and then call this elem_delete function. Atomicity
	 * relies on the caller potentially re-acquiring the lock or careful
	 * coordination if lock is external (which contradicts
	 * should_lock=false). This function only performs the pop step.
	 * @param key Ignored (must be NULL conceptually for stack).
	 * @return 0 on success. Negative error code on failure (-ENOENT if
	 * empty).
	 */
	long elem_delete(const void *key);

	/**
	 * @brief Get next key. Not applicable for Stacks.
	 * @param key Ignored.
	 * @param next_key Ignored.
	 * @return Always returns -EINVAL.
	 */
	int map_get_next_key(const void *key, void *next_key);

	// --- New methods mapping to bpf_map helper functions from Linux kernel
	// patch ---

	/**
	 * @brief Implements bpf_map_push_elem helper function.
	 * Adds an element to the top of the stack according to flags.
	 * @param value Pointer to the value data to push. Must not be NULL.
	 * @param flags BPF_EXIST: if full, remove oldest element and add new
	 * one. 0 or BPF_ANY: fail if full.
	 * @return 0 on success. Negative error code on failure.
	 */
	long map_push_elem(const void *value, uint64_t flags);

	/**
	 * @brief Implements bpf_map_pop_elem helper function.
	 * Removes an element from the top of the stack and copies its value.
	 * @param value Pointer to buffer where the popped value will be copied.
	 * @return 0 on success. Negative error code on failure (-ENOENT if
	 * empty).
	 */
	long map_pop_elem(void *value);

	/**
	 * @brief Implements bpf_map_peek_elem helper function.
	 * Gets an element from the top of the stack without removing it.
	 * @param value Pointer to buffer where the peeked value will be copied.
	 * @return 0 on success. Negative error code on failure (-ENOENT if
	 * empty).
	 */
	long map_peek_elem(void *value);

	// --- Helper methods ---
	unsigned int get_value_size() const;
	unsigned int get_max_entries() const;
	size_t get_current_size() const; // Calculates current number of
					 // elements
};

} // namespace bpftime

#endif // BPFTIME_STACK_MAP_HPP