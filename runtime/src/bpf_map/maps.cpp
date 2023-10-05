#include <bpf_map/maps.hpp>
#include <algorithm>
#include <array>
#include <boost/container/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <iostream>
#include <string>
#include <utility>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <bpf_map/array_map.hpp>
#include <bpf_map/hash_map.hpp>
#include <bpf_map/ringbuf_map.hpp>

using boost::interprocess::interprocess_sharable_mutex;
using boost::interprocess::scoped_lock;
using boost::interprocess::sharable_lock;
namespace bpftime
{

const void *bpf_map_handler::map_lookup_elem(const void *key) const
{
	const auto do_lookup = [&](auto *impl) -> const void * {
		if (impl->should_lock) {
			sharable_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_lookup(key);
		} else {
			return impl->elem_lookup(key);
		}
	};
	switch (type) {
	case BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_lookup(impl);
	}
	case BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_lookup(impl);
	}
	case BPF_MAP_TYPE_RINGBUF: {
		{
			auto impl = static_cast<ringbuf_map_impl *>(
				map_impl_ptr.get());
			return do_lookup(impl);
		}
	}
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_update_elem(const void *key, const void *value,
				      uint64_t flags) const
{
	const auto do_update = [&](auto *impl) -> long {
		if (impl->should_lock) {
			scoped_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_update(key, value, flags);
		} else {
			return impl->elem_update(key, value, flags);
		}
	};
	switch (type) {
	case BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	case BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	case BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::bpf_map_get_next_key(const void *key, void *next_key) const
{
	const auto do_get_next_key = [&](auto *impl) -> int {
		if (impl->should_lock) {
			sharable_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->bpf_map_get_next_key(key, next_key);
		} else {
			return impl->bpf_map_get_next_key(key, next_key);
		}
	};
	switch (type) {
	case BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_delete_elem(const void *key) const
{
	const auto do_delete = [&](auto *impl) -> long {
		if (impl->should_lock) {
			scoped_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_delete(key);
		} else {
			return impl->elem_delete(key);
		}
	};
	switch (type) {
	case BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	case BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	case BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::map_init(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		map_impl_ptr = memory.construct<hash_map_impl>(
			container_name.c_str())(memory, key_size, value_size);
		return 0;
	case BPF_MAP_TYPE_ARRAY:
		map_impl_ptr = memory.construct<array_map_impl>(
			container_name.c_str())(memory, value_size,
						max_entries);
		return 0;
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

void bpf_map_handler::map_free(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		memory.destroy<hash_map_impl>(container_name.c_str());
		break;
	case BPF_MAP_TYPE_ARRAY:
		memory.destroy<array_map_impl>(container_name.c_str());
		break;
	default:
		assert(false || "Unsupported map type");
	}
	map_impl_ptr = nullptr;
	return;
}

} // namespace bpftime
