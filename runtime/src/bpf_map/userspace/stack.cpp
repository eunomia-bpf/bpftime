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
	: _value_size(value_size), _max_entries(max_entries),
	  data(byte_allocator(memory.get_segment_manager()))
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Stack map value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::invalid_argument(
			"Stack map value_size or max_entries cannot be zero");
	}

	// Reserve space for maximum entries
	data.reserve(max_entries * _value_size);

	SPDLOG_DEBUG("Stack map constructed: value_size={}, max_entries={}",
		     _value_size, _max_entries);
}

// --- Internal helper implementations ---
size_t stack_map_impl::get_current_size() const
{
	return data.size() / _value_size;
}

bool stack_map_impl::is_full() const
{
	return get_current_size() >= _max_entries;
}

bool stack_map_impl::is_empty() const
{
	return data.empty();
}

// --- Methods mapping to standard BPF syscall commands ---

// Peek operation (elem_lookup - returns internal pointer)
void *stack_map_impl::elem_lookup(const void *key)
{
	// key is ignored for stack peek operation.
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack peek (lookup) failed: stack empty");
		return nullptr;
	}

	// Return pointer to the top element (last element in vector)
	size_t top_offset = (get_current_size() - 1) * _value_size;
	void *elem_ptr = data.data() + top_offset;
	SPDLOG_TRACE("Stack peek (lookup) success: returning ptr={:p}",
		     elem_ptr);
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
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Stack is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Remove oldest element (bottom of stack)
			SPDLOG_TRACE(
				"Stack elem_update (BPF_EXIST): stack full, removing oldest element");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			// flags == 0 or BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Stack elem_update: failed, stack full (size={})",
				get_current_size());
			return -E2BIG;
		}
	}

	// Add new element to the top of the stack (end of vector)
	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Stack elem_update: success, new size={}",
		     get_current_size());
	return 0;
}

// Delete operation (elem_delete - removes top element)
long stack_map_impl::elem_delete(const void *key)
{
	// key is ignored for stack
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	if (is_empty()) {
		SPDLOG_TRACE("Stack elem_delete failed: stack empty");
		return -ENOENT;
	}

	// Remove the top element (last element in vector)
	data.erase(data.end() - _value_size, data.end());
	SPDLOG_TRACE("Stack elem_delete success: new size={}",
		     get_current_size());
	return 0;
}

// Get next key - not supported for stacks
int stack_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Not supported for stack maps
	return -EINVAL;
}

// --- New methods mapping to bpf_map helper functions ---

// Push operation for bpf_map_push_elem helper
long stack_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == NULL) {
		SPDLOG_WARN(
			"Stack map_push_elem failed: value pointer is NULL");
		return -EINVAL;
	}

	// Validate flags
	if (flags != 0 && flags != BPF_ANY && flags != BPF_EXIST) {
		SPDLOG_WARN("Stack map_push_elem failed: invalid flags ({})",
			    flags);
		return -EINVAL;
	}

	// Lock for thread/process safety
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>
		lock(mutex);

	bool full = is_full();

	if (full) {
		// Stack is full, check flags
		if (flags == BPF_EXIST) {
			// BPF_EXIST: Remove oldest element (bottom of stack)
			SPDLOG_TRACE(
				"Stack map_push_elem (BPF_EXIST): stack full, removing oldest element");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			// flags == 0 or BPF_ANY: Fail if full
			SPDLOG_TRACE(
				"Stack map_push_elem: failed, stack full (size={})",
				get_current_size());
			return -E2BIG;
		}
	}

	// Add new element to the top of the stack (end of vector)
	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Stack map_push_elem: success, new size={}",
		     get_current_size());
	return 0;
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
		return -ENOENT;
	}

	// Copy the top element to the output buffer
	size_t top_offset = (get_current_size() - 1) * _value_size;
	memcpy(value, data.data() + top_offset, _value_size);

	// Remove the top element from the stack
	data.erase(data.end() - _value_size, data.end());

	SPDLOG_TRACE("Stack map_pop_elem success: new size={}",
		     get_current_size());
	return 0;
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
		return -ENOENT;
	}

	// Copy the top element to the output buffer
	size_t top_offset = (get_current_size() - 1) * _value_size;
	memcpy(value, data.data() + top_offset, _value_size);

	SPDLOG_TRACE("Stack map_peek_elem success");
	return 0;
}

// --- Helper method implementations ---
unsigned int stack_map_impl::get_value_size() const
{
	return _value_size;
}

unsigned int stack_map_impl::get_max_entries() const
{
	return _max_entries;
}

} // namespace bpftime