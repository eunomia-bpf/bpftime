/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, Your Name/Org Here (Adapt as needed)
 * All rights reserved.
 */
#include <bpf_map/userspace/queue.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bpftime
{

// Constructor
queue_map_impl::queue_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	uint32_t value_size, uint32_t max_entries)
	: head(0), tail(0), _value_size(value_size),
	  _max_entries(max_entries), capacity(max_entries + 1),
	  buffer_allocator(memory.get_segment_manager())
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Queue map value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::runtime_error(
			"Queue map value_size or max_entries cannot be zero");
	}
	try {
		// Allocate buffer in shared memory
		buffer = buffer_allocator.allocate(capacity * _value_size);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR(
			"Failed to allocate buffer for queue map ({} bytes): {}",
			capacity * _value_size, ex.what());
		throw std::runtime_error(
			"Failed to allocate shared memory buffer for queue map");
	}
    if (!buffer) {
         SPDLOG_ERROR(
			"Failed to allocate buffer for queue map ({} bytes) - allocate returned null",
			capacity * _value_size);
		throw std::runtime_error(
			"Failed to allocate buffer for queue map - null pointer");
	}
	SPDLOG_DEBUG(
		"Queue map constructed: value_size={}, max_entries={}, capacity={}, buffer_ptr={:p}",
		_value_size, _max_entries, capacity, (void *)buffer.get());
}

// Destructor
queue_map_impl::~queue_map_impl()
{
	SPDLOG_DEBUG("Destroying queue map: buffer_ptr={:p}, size={} bytes",
		     (void *)buffer.get(), capacity * _value_size);
	if (buffer) {
		try {
			// Deallocate the buffer using the stored allocator
			buffer_allocator.deallocate(buffer.get(),
						    capacity * _value_size);
		} catch (const std::exception &ex) {
			SPDLOG_ERROR("Exception during queue map buffer deallocation: {}", ex.what());
		} catch (...) {
			SPDLOG_ERROR("Unknown exception during queue map buffer deallocation");
		}
	}
}

// --- Internal helper implementations ---
bool queue_map_impl::is_full() const
{
	// Assumes called under lock or head/tail access is safe
	return (tail + 1) % capacity == head;
}

bool queue_map_impl::is_empty() const
{
	// Assumes called under lock or head/tail access is safe
	return head == tail;
}

// Internal dequeue - Must be called under lock
void queue_map_impl::internal_dequeue()
{
	// Caller must ensure queue is not empty
	head = (head + 1) % capacity;
}

// Internal enqueue - Must be called under lock
void queue_map_impl::internal_enqueue(const void *value)
{
	// Caller must ensure queue is not full
	void *dest_ptr = buffer.get() + tail * _value_size;
	memcpy(dest_ptr, value, _value_size);
	tail = (tail + 1) % capacity;
}


// --- Methods mapping to standard BPF syscall commands (within signature limits) ---

// Peek operation (elem_lookup - returns internal pointer)
void *queue_map_impl::elem_lookup(const void *key)
{
	// key is ignored for queue peek operation.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue peek (lookup) failed: queue empty");
		return nullptr; // Standard BPF lookup returns NULL on miss/failure.
	}

	// Calculate pointer to the element at the head
	void *elem_ptr = buffer.get() + head * _value_size;
	SPDLOG_TRACE("Queue peek (lookup) success: head={}, returning ptr={:p}", head, elem_ptr);
	// Return direct pointer. Caller MUST copy data from this pointer.
	return elem_ptr;
}

// Push operation (elem_update - with flags)
long queue_map_impl::elem_update(const void *key, const void *value, uint64_t flags)
{
	// key is ignored for queue, should be NULL
	if (key != NULL) {
		SPDLOG_WARN("Queue push (update) called with non-NULL key, ignoring key.");
	}
	if (value == NULL) {
        SPDLOG_WARN("Queue push (update) failed: value pointer is NULL");
		return -EINVAL; // Cannot push NULL value
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);

	bool full = is_full();

	if (full) {
		// Queue is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Overwrite oldest element
			SPDLOG_TRACE("Queue push (BPF_EXIST): queue full, removing oldest element at head={}", head);
			internal_dequeue(); // Make space by removing the head element
			internal_enqueue(value); // Add the new element at the tail
			SPDLOG_TRACE("Queue push (BPF_EXIST): success, new tail={}", tail);
			return 0; // Success
		} else if (flags == BPF_ANY) {
			// BPF_ANY: Fail if full
			SPDLOG_TRACE("Queue push (BPF_ANY): failed, queue full (head={}, tail={}, capacity={})", head, tail, capacity);
			return -EBUSY;
		} else {
			// Invalid flags for push operation
			SPDLOG_WARN("Queue push: invalid flags value: {}", flags);
			return -EINVAL;
		}
	} else {
		// Queue is not full, standard enqueue allowed for valid flags
		if (flags != BPF_ANY && flags != BPF_EXIST) {
             SPDLOG_WARN("Queue push: invalid flags value: {}", flags);
             return -EINVAL;
        }
		// Perform the enqueue
		internal_enqueue(value);
		SPDLOG_TRACE("Queue push (BPF_ANY/EXIST): success, queue not full, new tail={}", tail);
		return 0; // Success
	}
}

// Dequeue operation (elem_delete - delete part only)
long queue_map_impl::elem_delete(const void *key)
{
	// key is ignored for queue dequeue.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue pop (delete part) failed: queue empty");
		return -ENOENT; // No element exists to be deleted
	}

	// Dequeue simply advances the head pointer.
	uint32_t old_head = head;
	internal_dequeue(); // Use the internal helper
	SPDLOG_TRACE("Queue pop (delete part) success: old_head={}, new_head={}", old_head, head);

	// NOTE: This function ONLY performs the delete. The caller (bpftime handler)
	// is responsible for calling elem_lookup first and copying the value
	// to implement the full BPF_MAP_LOOKUP_AND_DELETE_ELEM semantic.
	return 0; // Success
}

// Get next key - Not applicable
int queue_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_TRACE("Queue map_get_next_key called (operation not supported)");
	return -EINVAL;
}


// --- Helper method implementations ---
uint32_t queue_map_impl::get_value_size() const
{
	// No lock needed, immutable
	return _value_size;
}

uint32_t queue_map_impl::get_max_entries() const
{
	// No lock needed, immutable
	return _max_entries;
}

size_t queue_map_impl::get_current_size()
{
	// Lock to get consistent head/tail
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);
	if (tail >= head) {
		return tail - head;
	} else {
		// Tail has wrapped around
		return capacity - head + tail;
	}
}

} // namespace bpftime