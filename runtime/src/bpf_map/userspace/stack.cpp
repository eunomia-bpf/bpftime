#include <bpf_map/userspace/stack.hpp>
#include <spdlog/spdlog.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bpftime
{

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

	data.reserve(max_entries * _value_size);

	SPDLOG_DEBUG("Stack map constructed: value_size={}, max_entries={}",
		     _value_size, _max_entries);
}

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

void *stack_map_impl::elem_lookup(const void *key)
{
	if (is_empty()) {
		SPDLOG_TRACE("Stack peek (lookup) failed: stack empty");
		errno = ENOENT;
		return nullptr;
	}

	size_t top_offset = (get_current_size() - 1) * _value_size;
	void *elem_ptr = data.data() + top_offset;
	SPDLOG_TRACE("Stack peek (lookup) success: returning ptr={:p}",
		     elem_ptr);
	return elem_ptr;
}

long stack_map_impl::elem_update(const void *key, const void *value,
				 uint64_t flags)
{
	if (key != nullptr) {
		SPDLOG_WARN(
			"Stack push (update) called with non-nullptr key, ignoring key.");
	}
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Stack push (update) failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	bool full = is_full();

	if (full) {
		if (flags == BPF_EXIST) {
			SPDLOG_TRACE(
				"Stack elem_update (BPF_EXIST): stack full, removing oldest element (bottom)");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			SPDLOG_TRACE(
				"Stack elem_update: failed, stack full (size={})",
				get_current_size());
			errno = E2BIG;
			return -1;
		}
	}

	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Stack elem_update: success, new size={}",
		     get_current_size());
	return 0;
}

long stack_map_impl::elem_delete(const void *key)
{
	if (is_empty()) {
		SPDLOG_TRACE("Stack elem_delete failed: stack empty");
		errno = ENOENT;
		return -1;
	}

	data.erase(data.end() - _value_size, data.end());
	SPDLOG_TRACE("Stack elem_delete success: new size={}",
		     get_current_size());
	return 0;
}

int stack_map_impl::map_get_next_key(const void *key, void *next_key)
{
	errno = EINVAL;
	return -1;
}

long stack_map_impl::map_push_elem(const void *value, uint64_t flags)
{
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Stack map_push_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (flags != 0 && flags != BPF_ANY && flags != BPF_EXIST) {
		SPDLOG_ERROR("Stack map_push_elem failed: invalid flags ({})",
			     flags);
		errno = EINVAL;
		return -1;
	}

	bool full = is_full();

	if (full) {
		if (flags == BPF_EXIST) {
			SPDLOG_TRACE(
				"Stack map_push_elem (BPF_EXIST): stack full, removing oldest element (bottom)");
			data.erase(data.begin(), data.begin() + _value_size);
		} else {
			SPDLOG_TRACE(
				"Stack map_push_elem: failed, stack full (size={})",
				get_current_size());
			errno = E2BIG;
			return -1;
		}
	}

	const uint8_t *value_bytes = static_cast<const uint8_t *>(value);
	data.insert(data.end(), value_bytes, value_bytes + _value_size);

	SPDLOG_TRACE("Stack map_push_elem: success, new size={}",
		     get_current_size());
	return 0;
}

long stack_map_impl::map_pop_elem(void *value)
{
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Stack map_pop_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (is_empty()) {
		SPDLOG_TRACE("Stack map_pop_elem failed: stack empty");
		errno = ENOENT;
		return -1;
	}

	size_t top_offset = (get_current_size() - 1) * _value_size;
	memcpy(value, data.data() + top_offset, _value_size);

	data.erase(data.end() - _value_size, data.end());

	SPDLOG_TRACE("Stack map_pop_elem success: new size={}",
		     get_current_size());
	return 0;
}

long stack_map_impl::map_peek_elem(void *value)
{
	if (value == nullptr) {
		SPDLOG_ERROR(
			"Stack map_peek_elem failed: value pointer is nullptr");
		errno = EINVAL;
		return -1;
	}

	if (is_empty()) {
		SPDLOG_TRACE("Stack map_peek_elem failed: stack empty");
		errno = ENOENT;
		return -1;
	}

	size_t top_offset = (get_current_size() - 1) * _value_size;
	memcpy(value, data.data() + top_offset, _value_size);

	SPDLOG_TRACE("Stack map_peek_elem success");
	return 0;
}

unsigned int stack_map_impl::get_value_size() const
{
	return _value_size;
}

unsigned int stack_map_impl::get_max_entries() const
{
	return _max_entries;
}

} // namespace bpftime