/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, Your Name/Org Here (Adapt as needed)
 * All rights reserved.
 */
#include <bpf_map/userspace/stack.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bpftime
{

// Constructor
stack_map_impl::stack_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	unsigned int value_size, unsigned int max_entries)
	: top(0), _value_size(value_size), _max_entries(max_entries),
	  capacity(max_entries), buffer_allocator(memory.get_segment_manager())
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Stack map value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::runtime_error(
			"Stack map value_size or max_entries cannot be zero");
	}
	try {
		// Allocate buffer in shared memory
		buffer = buffer_allocator.allocate(capacity * _value_size);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR(
			"Failed to allocate buffer for stack map ({} bytes): {}",
			capacity * _value_size, ex.what());
		throw std::runtime_error(
			"Failed to allocate shared memory buffer for stack map");
	}
	if (!buffer) {
		SPDLOG_ERROR(
			"Failed to allocate buffer for stack map ({} bytes) - allocate returned null",
			capacity * _value_size);
		throw std::runtime_error(
			"Failed to allocate buffer for stack map - null pointer");
	}
	SPDLOG_DEBUG(
		"Stack map constructed: value_size={}, max_entries={}, capacity={}, buffer_ptr={:p}",
		_value_size, _max_entries, capacity, (void *)buffer.get());
}

// Destructor
stack_map_impl::~stack_map_impl()
{
	SPDLOG_DEBUG("Destroying stack map: buffer_ptr={:p}, size={} bytes",
		     (void *)buffer.get(), capacity * _value_size);
	if (buffer) {
		try {
			// Deallocate the buffer using the stored allocator
			buffer_allocator.deallocate(buffer.get(),
						    capacity * _value_size);
		} catch (const std::exception &ex) {
			SPDLOG_ERROR(
				"Exception during stack map buffer deallocation: {}",
				ex.what());
		} catch (...) {
			SPDLOG_ERROR(
				"Unknown exception during stack map buffer deallocation");
		}
	}
}

// --- Internal helper implementations ---
bool stack_map_impl::is_full() const
{
	// Assumes called under lock or top access is safe
	return top == capacity;
}

bool stack_map_impl::is_empty() const
{
	// Assumes called under lock or top access is safe
	return top == 0;
}

// Internal pop - Must be called under lock
void stack_map_impl::internal_pop()
{
	// Caller must ensure stack is not empty
	top--;
}

// Internal push - Must be called under lock
void stack_map_impl::internal_push(const void *value)
{
	// Caller must ensure stack is not full
	void *dest_ptr = buffer.get() + top * _value_size;
	memcpy(dest_ptr, value, _value_size);
	top++;
}

// --- Methods mapping to standard BPF syscall commands (within signature
// limits) ---

// Peek operation (elem_lookup - returns internal pointer)
void *stack_map_impl::elem_lookup(const void *key)
{
	// key is ignored for stack peek operation.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack peek (lookup) failed: stack empty");
		return nullptr; // Standard BPF lookup returns NULL on
				// miss/failure.
	}

	// Calculate pointer to the element at the top of the stack
	void *elem_ptr = buffer.get() + (top - 1) * _value_size;
	SPDLOG_TRACE("Stack peek (lookup) success: top={}, returning ptr={:p}",
		     top, elem_ptr);
	// Return direct pointer. Caller MUST copy data from this pointer.
	return elem_ptr;
}

// Push operation (elem_update - with flags)
long stack_map_impl::elem_update(const void *key, const void *value,
				 uint64_t flags)
{
	// key is ignored for stack, should be NULL
	if (key != NULL) {
		SPDLOG_WARN(
			"Stack push (update) called with non-NULL key, ignoring key.");
	}
	if (value == NULL) {
		SPDLOG_WARN(
			"Stack push (update) failed: value pointer is NULL");
		return -EINVAL; // Cannot push NULL value
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Stack is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Overwrite oldest element (bottom of stack)
			// For stack, we need to shift all elements down and add
			// new one at top
			SPDLOG_TRACE(
				"Stack push (BPF_EXIST): stack full, removing oldest element at bottom");
			// Shift all elements down by one position (remove
			// bottom element)
			memmove(buffer.get(), buffer.get() + _value_size,
				(capacity - 1) * _value_size);
			// Add new element at the top (which is now at position
			// capacity-1)
			void *dest_ptr =
				buffer.get() + (capacity - 1) * _value_size;
			memcpy(dest_ptr, value, _value_size);
			// top remains the same (capacity)
			SPDLOG_TRACE(
				"Stack push (BPF_EXIST): success, top remains={}",
				top);
			return 0; // Success
		} else if (flags == BPF_ANY) {
			// BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Stack push (BPF_ANY): failed, stack full (top={}, capacity={})",
				top, capacity);
			return -EBUSY;
		} else {
			// Invalid flags for push operation
			SPDLOG_WARN("Stack push: invalid flags value: {}",
				    flags);
			return -EINVAL;
		}
	} else {
		// Stack is not full, standard push allowed for valid flags
		if (flags != BPF_ANY && flags != BPF_EXIST) {
			SPDLOG_WARN("Stack push: invalid flags value: {}",
				    flags);
			return -EINVAL;
		}
		// Perform the push
		internal_push(value);
		SPDLOG_TRACE(
			"Stack push (BPF_ANY/EXIST): success, stack not full, new top={}",
			top);
		return 0; // Success
	}
}

// Pop operation (elem_delete - delete part only)
long stack_map_impl::elem_delete(const void *key)
{
	// key is ignored for stack pop.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack pop (delete part) failed: stack empty");
		return -ENOENT; // No element exists to be deleted
	}

	// Pop simply decrements the top pointer.
	unsigned int old_top = top;
	internal_pop(); // Use the internal helper
	SPDLOG_TRACE("Stack pop (delete part) success: old_top={}, new_top={}",
		     old_top, top);

	// NOTE: This function ONLY performs the delete. The caller (bpftime
	// handler) is responsible for calling elem_lookup first and copying the
	// value to implement the full BPF_MAP_LOOKUP_AND_DELETE_ELEM semantic.
	return 0; // Success
}

// Get next key - Not applicable
int stack_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_TRACE("Stack map_get_next_key called (operation not supported)");
	return -EINVAL;
}

// --- New helper function implementations mapping to Linux kernel patch ---

// Push operation for bpf_map_push_elem helper
long stack_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == NULL) {
		SPDLOG_WARN(
			"Stack map_push_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Stack is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Overwrite oldest element (bottom of stack)
			SPDLOG_TRACE(
				"Stack map_push_elem (BPF_EXIST): stack full, removing oldest element at bottom");
			// Shift all elements down by one position (remove
			// bottom element)
			memmove(buffer.get(), buffer.get() + _value_size,
				(capacity - 1) * _value_size);
			// Add new element at the top (which is now at position
			// capacity-1)
			void *dest_ptr =
				buffer.get() + (capacity - 1) * _value_size;
			memcpy(dest_ptr, value, _value_size);
			// top remains the same (capacity)
			SPDLOG_TRACE(
				"Stack map_push_elem (BPF_EXIST): success, top remains={}",
				top);
			return 0; // Success
		} else {
			// flags == 0 or BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Stack map_push_elem: failed, stack full (top={}, capacity={})",
				top, capacity);
			return -E2BIG;
		}
	} else {
		// Stack is not full, perform the push for any valid flags
		internal_push(value);
		SPDLOG_TRACE(
			"Stack map_push_elem: success, stack not full, new top={}",
			top);
		return 0; // Success
	}
}

// Pop operation for bpf_map_pop_elem helper
long stack_map_impl::map_pop_elem(void *value)
{
	if (value == NULL) {
		SPDLOG_WARN("Stack map_pop_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack map_pop_elem failed: stack empty");
		return -ENOENT; // No element exists to be popped
	}

	// Copy the element at the top to the output buffer
	void *elem_ptr = buffer.get() + (top - 1) * _value_size;
	memcpy(value, elem_ptr, _value_size);

	// Remove the element from the stack
	unsigned int old_top = top;
	internal_pop();
	SPDLOG_TRACE(
		"Stack map_pop_elem success: copied and removed element at old_top={}, new_top={}",
		old_top, top);

	return 0; // Success
}

// Peek operation for bpf_map_peek_elem helper
long stack_map_impl::map_peek_elem(void *value)
{
	if (value == NULL) {
		SPDLOG_WARN(
			"Stack map_peek_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack map_peek_elem failed: stack empty");
		return -ENOENT; // No element exists to be peeked
	}

	// Copy the element at the top to the output buffer
	void *elem_ptr = buffer.get() + (top - 1) * _value_size;
	memcpy(value, elem_ptr, _value_size);
	SPDLOG_TRACE("Stack map_peek_elem success: copied element at top={}",
		     top);

	return 0; // Success
}

// --- Helper method implementations ---
unsigned int stack_map_impl::get_value_size() const
{
	// No lock needed, immutable
	return _value_size;
}

unsigned int stack_map_impl::get_max_entries() const
{
	// No lock needed, immutable
	return _max_entries;
}

size_t stack_map_impl::get_current_size()
{
	// Lock to get consistent top
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);
	return top;
}

} // namespace bpftime