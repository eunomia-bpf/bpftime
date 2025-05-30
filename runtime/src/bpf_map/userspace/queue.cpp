#include <bpf_map/userspace/queue.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bpftime
{

queue_map_impl::queue_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	unsigned int value_size, unsigned int max_entries)
	: _value_size(value_size), _max_entries(max_entries),
	  data(byte_allocator(memory.get_segment_manager())), head(0), tail(0),
	  count(0)
{
	if (value_size == 0 || max_entries == 0) {
		SPDLOG_ERROR(
			"Queue map value_size ({}) or max_entries ({}) cannot be zero",
			value_size, max_entries);
		throw std::invalid_argument(
			"Queue map value_size or max_entries cannot be zero");
	}

	data.resize(max_entries * _value_size);

	SPDLOG_DEBUG("Queue map constructed: value_size={}, max_entries={}",
		     _value_size, _max_entries);
}

size_t queue_map_impl::get_current_size() const
{
	return count;
}

bool queue_map_impl::is_full() const
{
	return count >= _max_entries;
}

bool queue_map_impl::is_empty() const
{
	return count == 0;
}

unsigned int queue_map_impl::next_index(unsigned int index) const
{
	return (index + 1) % _max_entries;
}

void *queue_map_impl::elem_lookup(const void *key)
{
	if (is_empty()) {
		SPDLOG_TRACE("Queue peek (lookup) failed: queue empty");
		errno = ENOENT;
		return nullptr;
	}

	void *elem_ptr = data.data() + head * _value_size;
	SPDLOG_TRACE("Queue peek (lookup) success: returning ptr={:p}",
		     elem_ptr);
	return elem_ptr;
}

long queue_map_impl::elem_update(const void *key, const void *value,
				 uint64_t flags)
{
	if (key != nullptr) {
		SPDLOG_WARN(
			"Queue push (update) called with non-nullptr key, ignoring key.");
	}
	if (value == nullptr) {
		SPDLOG_WARN(
			"Queue push (update) failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	bool full = is_full();

	if (full) {
		if (flags == BPF_EXIST) {
			SPDLOG_TRACE(
				"Queue elem_update (BPF_EXIST): queue full, removing oldest element");
			head = next_index(head);
			count--;
		} else {
			SPDLOG_TRACE(
				"Queue elem_update: failed, queue full (size={})",
				get_current_size());
			errno = E2BIG;
			return -1;
		}
	}

	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	memcpy(data.data() + tail * _value_size, value_bytes, _value_size);
	tail = next_index(tail);
	count++;

	SPDLOG_TRACE("Queue elem_update: success, new size={}",
		     get_current_size());
	return 0;
}

long queue_map_impl::elem_delete(const void *key)
{
	if (is_empty()) {
		SPDLOG_TRACE("Queue elem_delete failed: queue empty");
		errno = ENOENT;
		return -1;
	}

	head = next_index(head);
	count--;
	SPDLOG_TRACE("Queue elem_delete success: new size={}",
		     get_current_size());
	return 0;
}

int queue_map_impl::map_get_next_key(const void *key, void *next_key)
{
	errno = EINVAL;
	return -1;
}

long queue_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == nullptr) {
		SPDLOG_WARN(
			"Queue map_push_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (flags != 0 && flags != BPF_ANY && flags != BPF_EXIST) {
		SPDLOG_WARN("Queue map_push_elem failed: invalid flags ({})",
			    flags);
		errno = EINVAL;
		return -1;
	}

	bool full = is_full();

	if (full) {
		if (flags == BPF_EXIST) {
			SPDLOG_TRACE(
				"Queue map_push_elem (BPF_EXIST): queue full, removing oldest element");
			head = next_index(head);
			count--;
		} else {
			SPDLOG_TRACE(
				"Queue map_push_elem: failed, queue full (size={})",
				get_current_size());
			errno = E2BIG;
			return -1;
		}
	}

	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	memcpy(data.data() + tail * _value_size, value_bytes, _value_size);
	tail = next_index(tail);
	count++;

	SPDLOG_TRACE("Queue map_push_elem: success, new size={}",
		     get_current_size());
	return 0;
}

long queue_map_impl::map_pop_elem(void *value)
{
	if (value == nullptr) {
		SPDLOG_WARN(
			"Queue map_pop_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (is_empty()) {
		SPDLOG_TRACE("Queue map_pop_elem failed: queue empty");
		errno = ENOENT;
		return -1;
	}

	memcpy(value, data.data() + head * _value_size, _value_size);

	head = next_index(head);
	count--;

	SPDLOG_TRACE("Queue map_pop_elem success: new size={}",
		     get_current_size());
	return 0;
}

long queue_map_impl::map_peek_elem(void *value)
{
	if (value == nullptr) {
		SPDLOG_WARN(
			"Queue map_peek_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (is_empty()) {
		SPDLOG_TRACE("Queue map_peek_elem failed: queue empty");
		errno = ENOENT;
		return -1;
	}

	memcpy(value, data.data() + head * _value_size, _value_size);

	SPDLOG_TRACE("Queue map_peek_elem success");
	return 0;
}

unsigned int queue_map_impl::get_value_size() const
{
	return _value_size;
}

unsigned int queue_map_impl::get_max_entries() const
{
	return _max_entries;
}

} // namespace bpftime