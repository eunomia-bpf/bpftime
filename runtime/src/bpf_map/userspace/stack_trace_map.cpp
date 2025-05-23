#include "bpf_map/map_common_def.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <boost/container_hash/hash_fwd.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "stack_trace_map.hpp"
using namespace bpftime;

stack_trace_map_impl::stack_trace_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries)
	: max_entries(max_entries),
	  data(max_entries, uint32_hasher(), std::equal_to<uint32_t>(),
	       stack_trace_map::map_allocator(memory.get_segment_manager())),
	  key_buf(value_size / 8, memory.get_segment_manager())
{
	if (key_size != 4) {
		SPDLOG_ERROR("Key size of stack trace map must be 4");
		throw std::runtime_error(
			"Key size of stack trace map must be 4");
	}
	if (value_size % 8 != 0) {
		SPDLOG_ERROR(
			"value_size of stack_trace_map must be a multiple of 8, unexpected: {}",
			value_size);
		throw std::runtime_error("Invalid value_size");
	}
	max_stack_entries = value_size / 8;
}

void *stack_trace_map_impl::elem_lookup(const void *key)
{
	if (key == nullptr) {
		errno = EINVAL;
		SPDLOG_ERROR(
			"You can't lookup a stack trace map with NULL key");
		return nullptr;
	}
	uint32_t index = *(int *)key;
	auto itr = this->data.find(index);
	if (itr == data.end()) {
		errno = ENOENT;
		return nullptr;
	}
	return itr->second.data();
}

long stack_trace_map_impl::elem_update(const void *key, const void *value,
				       uint64_t flags)
{
	SPDLOG_ERROR("You can't update a stack trace map");
	return -ENOTSUP;
}

long stack_trace_map_impl::elem_delete(const void *key)
{
	if (key == nullptr) {
		SPDLOG_ERROR(
			"You can't lookup a stack trace map with NULL key");
		errno = EINVAL;
		return -1;
	}
	uint32_t index = *(int *)key;
	if (auto itr = data.find(index); itr != data.end()) {
		data.erase(itr);
	} else {
		errno = ENOENT;
		return -1;
	}

	return 0;
}

int stack_trace_map_impl::map_get_next_key(const void *key, void *next_key)
{
	errno = ENOTSUP;
	SPDLOG_ERROR("get_next_key is not supported by stack_trace_map");
	return -1;
}

int stack_trace_map_impl::fill_stack_trace(const std::vector<uint64_t> &stk,
					   bool discard_old_one,
					   bool compare_only_by_hash)
{
	auto to_put_hash =
		stack_trace_map::hash_stack_trace(stk) % this->max_entries;
	auto itr = this->data.find(to_put_hash);
	key_buf.resize(stk.size());
	std::copy(stk.begin(), stk.end(), key_buf.begin());
	SPDLOG_DEBUG("Filling stack trace, to_put_hash={}, stack_trace_size={}",
		     to_put_hash, key_buf.size());
	if (itr == this->data.end()) {
		data.emplace(stack_trace_map::value_ty(to_put_hash, key_buf));
		SPDLOG_DEBUG("Set new stack map entry");
		return to_put_hash;
	} else {
		bool equals = true;
		if (!compare_only_by_hash) {
			equals = equals && itr->second.size() == stk.size();
			if (!equals) {
				equals = equals &&
					 std::equal(stk.begin(), stk.end(),
						    itr->second.begin());
			}
		}
		SPDLOG_DEBUG("equals={}", equals);
		if (equals) {
			SPDLOG_DEBUG("Early return due to same hash");
			return to_put_hash;
		}
		if (discard_old_one) {
			SPDLOG_DEBUG(
				"Discarding existsing stack trace entry..");
			itr->second = key_buf;
		}
		SPDLOG_DEBUG("Got hash {}", to_put_hash);
		return to_put_hash;
	}
}
