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
	unsigned int value_size, unsigned int max_entries)
	: _value_size(value_size), _max_entries(max_entries),
	  data(byte_allocator(memory.get_segment_manager()))
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Queue map value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::invalid_argument(
			"Queue map value_size or max_entries cannot be zero");
	}

	// Reserve space for maximum entries
	data.reserve(max_entries * _value_size);

	SPDLOG_DEBUG("Queue map constructed: value_size={}, max_entries={}",
		     _value_size, _max_entries);
}

// --- Internal helper implementations ---
size_t queue_map_impl::get_current_size() const
{
	return data.size() / _value_size;
}

bool queue_map_impl::is_full() const
{
	return get_current_size() >= _max_entries;
}

bool queue_map_impl::is_empty() const
{
	return data.empty();
}

// --- Methods mapping to standard BPF syscall commands ---

// Peek operation (elem_lookup - returns internal pointer)
void *queue_map_impl::elem_lookup(const void *key)
{
	// key is ignored for queue peek operation.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue peek (lookup) failed: queue empty");
		errno = ENOTSUP; // Set errno as expected by tests
		return nullptr;
	}

	// Return pointer to the first element (front of queue)
	void *elem_ptr = data.data();
	SPDLOG_TRACE("Queue peek (lookup) success: returning ptr={:p}",
		     elem_ptr);
	return elem_ptr;
}

// Push operation (elem_update - with flags)
long queue_map_impl::elem_update(const void *key, const void *value,
				 uint64_t flags)
{
	// key is ignored for queue, should be NULL
	if (key != NULL) {
		SPDLOG_WARN(
			"Queue push (update) called with non-NULL key, ignoring key.");
	}
	if (value == NULL) {
		SPDLOG_WARN(
			"Queue push (update) failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Queue is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Overwrite oldest element (remove from
			// front)
			SPDLOG_TRACE(
				"Queue elem_update (BPF_EXIST): queue full, removing oldest element");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			// flags == 0 or BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Queue elem_update: failed, queue full (size={})",
				get_current_size());
			return -E2BIG;
		}
	}

	// Add new element to the back of the queue
	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Queue elem_update: success, new size={}",
		     get_current_size());
	return 0;
}

// Delete operation (elem_delete - removes front element)
long queue_map_impl::elem_delete(const void *key)
{
	// key is ignored for queue
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue elem_delete failed: queue empty");
		return -ENOENT;
	}

	// Remove the front element
	data.erase(data.begin(), data.begin() + _value_size);
	SPDLOG_TRACE("Queue elem_delete success: new size={}",
		     get_current_size());
	return 0;
}

// Get next key - not supported for queues
int queue_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Not supported for queue maps
	return -EINVAL;
}

// --- New methods mapping to bpf_map helper functions ---

// Push operation for bpf_map_push_elem helper
long queue_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == NULL) {
		SPDLOG_WARN(
			"Queue map_push_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Validate flags
	if (flags != 0 && flags != BPF_ANY && flags != BPF_EXIST) {
		SPDLOG_WARN("Queue map_push_elem failed: invalid flags ({})",
			    flags);
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Queue is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Overwrite oldest element
			SPDLOG_TRACE(
				"Queue map_push_elem (BPF_EXIST): queue full, removing oldest element");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			// flags == 0 or BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Queue map_push_elem: failed, queue full (size={})",
				get_current_size());
			return -E2BIG;
		}
	}

	// Add new element to the back of the queue
	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Queue map_push_elem: success, new size={}",
		     get_current_size());
	return 0;
}

// Pop operation for bpf_map_pop_elem helper
long queue_map_impl::map_pop_elem(void *value)
{
	if (value == NULL) {
		SPDLOG_WARN("Queue map_pop_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue map_pop_elem failed: queue empty");
		return -ENOENT;
	}

	// Copy the front element to the output buffer
	memcpy(value, data.data(), _value_size);

	// Remove the front element from the queue
	data.erase(data.begin(), data.begin() + _value_size);

	SPDLOG_TRACE("Queue map_pop_elem success: new size={}",
		     get_current_size());
	return 0;
}

// Peek operation for bpf_map_peek_elem helper
long queue_map_impl::map_peek_elem(void *value)
{
	if (value == NULL) {
		SPDLOG_WARN(
			"Queue map_peek_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Queue map_peek_elem failed: queue empty");
		return -ENOENT;
	}

	// Copy the front element to the output buffer
	memcpy(value, data.data(), _value_size);

	SPDLOG_TRACE("Queue map_peek_elem success");
	return 0;
}

// --- Helper method implementations ---
unsigned int queue_map_impl::get_value_size() const
{
	return _value_size;
}

unsigned int queue_map_impl::get_max_entries() const
{
	return _max_entries;
}

} // namespace bpftime