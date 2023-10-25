#include "bpf_map/per_cpu_array_map.hpp"
#include "bpf_map/per_cpu_hash_map.hpp"
#include <bpf_map/perf_event_array_map.hpp>
#include "spdlog/spdlog.h"
#include <handler/map_handler.hpp>
#include <bpf_map/array_map.hpp>
#include <bpf_map/hash_map.hpp>
#include <bpf_map/ringbuf_map.hpp>
#include <bpf_map/array_map_kernel_user.hpp>
#include <bpf_map/hash_map_kernel_user.hpp>

using boost::interprocess::interprocess_sharable_mutex;
using boost::interprocess::scoped_lock;
using boost::interprocess::sharable_lock;

namespace bpftime
{
std::string bpf_map_handler::get_container_name()
{
	return "ebpf_map_fd_" + std::string(name.c_str());
}
uint32_t bpf_map_handler::get_value_size() const
{
	auto result = value_size;
	if ((type == bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY) ||
	    (type == bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH)) {
		result *= sysconf(_SC_NPROCESSORS_ONLN);
	}
	return result;
}
std::optional<ringbuf_map_impl *>
bpf_map_handler::try_get_ringbuf_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_RINGBUF)
		return {};
	return static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
}
std::optional<array_map_impl *> bpf_map_handler::try_get_array_map_impl() const
{
	if (type != bpf_map_type::BPF_MAP_TYPE_ARRAY)
		return {};
	return static_cast<array_map_impl *>(map_impl_ptr.get());
}

const void *bpf_map_handler::map_lookup_elem(const void *key,
					     bool from_userspace) const
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
	const auto do_lookup_userspace = [&](auto *impl) -> const void * {
		if (impl->should_lock) {
			sharable_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_lookup_userspace(key);
		} else {
			return impl->elem_lookup_userspace(key);
		}
	};

	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		auto impl = static_cast<perf_event_array_map_impl *>(
			map_impl_ptr.get());
		return do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		auto impl = static_cast<per_cpu_array_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_lookup_userspace(impl) :
					do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		auto impl = static_cast<per_cpu_hash_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_lookup_userspace(impl) :
					do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		auto impl = static_cast<array_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_lookup(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		auto impl = static_cast<hash_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_lookup(impl);
	}
	default:
		assert(false && "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_update_elem(const void *key, const void *value,
				      uint64_t flags, bool from_userspace) const
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

	const auto do_update_userspace = [&](auto *impl) -> long {
		if (impl->should_lock) {
			scoped_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_update_userspace(key, value, flags);
		} else {
			return impl->elem_update_userspace(key, value, flags);
		}
	};
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		auto impl = static_cast<perf_event_array_map_impl *>(
			map_impl_ptr.get());
		return do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		auto impl = static_cast<per_cpu_array_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_update_userspace(impl) :
					do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		auto impl = static_cast<per_cpu_hash_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_update_userspace(impl) :
					do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		auto impl = static_cast<array_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_update(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		auto impl = static_cast<hash_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_update(impl);
	}
	default:
		assert(false && "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::bpf_map_get_next_key(const void *key, void *next_key,
					  bool from_userspace) const
{
	const auto do_get_next_key = [&](auto *impl) -> int {
		if (impl->should_lock) {
			sharable_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->map_get_next_key(key, next_key);
		} else {
			return impl->map_get_next_key(key, next_key);
		}
	};
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		auto impl = static_cast<perf_event_array_map_impl *>(
			map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		auto impl = static_cast<per_cpu_array_map_impl *>(
			map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		auto impl = static_cast<per_cpu_hash_map_impl *>(
			map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		auto impl = static_cast<array_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		auto impl = static_cast<hash_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_get_next_key(impl);
	}
	default:
		assert(false && "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_delete_elem(const void *key,
				      bool from_userspace) const
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
	const auto do_delete_userspace = [&](auto *impl) -> long {
		if (impl->should_lock) {
			scoped_lock<interprocess_sharable_mutex> guard(
				*map_mutex);
			return impl->elem_delete_userspace(key);
		} else {
			return impl->elem_delete_userspace(key);
		}
	};

	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		auto impl = static_cast<hash_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		auto impl = static_cast<array_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto impl = static_cast<ringbuf_map_impl *>(map_impl_ptr.get());
		return do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		auto impl = static_cast<perf_event_array_map_impl *>(
			map_impl_ptr.get());
		return do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		auto impl = static_cast<per_cpu_array_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_delete_userspace(impl) :
					do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		auto impl = static_cast<per_cpu_hash_map_impl *>(
			map_impl_ptr.get());
		return from_userspace ? do_delete_userspace(impl) :
					do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		auto impl = static_cast<array_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_delete(impl);
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		auto impl = static_cast<hash_map_kernel_user_impl *>(
			map_impl_ptr.get());
		return do_delete(impl);
	}
	default:
		assert(false && "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::map_init(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH: {
		map_impl_ptr = memory.construct<hash_map_impl>(
			container_name.c_str())(memory, key_size, value_size);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_ARRAY: {
		map_impl_ptr = memory.construct<array_map_impl>(
			container_name.c_str())(memory, value_size,
						max_entries);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF: {
		auto max_ent = max_entries;
		int pop_cnt = 0;
		while (max_ent) {
			pop_cnt += (max_ent & 1);
			max_ent >>= 1;
		}
		if (pop_cnt != 1) {
			spdlog::error(
				"Failed to create ringbuf map, max_entries must be a power of 2, current: {}",
				max_entries);
			return -1;
		}
		map_impl_ptr = memory.construct<ringbuf_map_impl>(
			container_name.c_str())(max_entries, memory);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY: {
		map_impl_ptr = memory.construct<perf_event_array_map_impl>(
			container_name.c_str())(memory, key_size, value_size,
						max_entries);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY: {
		map_impl_ptr = memory.construct<per_cpu_array_map_impl>(
			container_name.c_str())(memory, value_size,
						max_entries);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH: {
		map_impl_ptr = memory.construct<per_cpu_hash_map_impl>(
			container_name.c_str())(memory, key_size, value_size);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_ARRAY: {
		map_impl_ptr = memory.construct<array_map_kernel_user_impl>(
			container_name.c_str())(memory, kernel_map_id);
		return 0;
	}
	case bpf_map_type::BPF_MAP_TYPE_KERNEL_USER_HASH: {
		map_impl_ptr = memory.construct<hash_map_kernel_user_impl>(
			container_name.c_str())(memory, kernel_map_id);
		return 0;
	}
	default:
		spdlog::error("Unsupported map type: {}", (int)type);
		assert(false && "Unsupported map type");
	}
	return 0;
}

void bpf_map_handler::map_free(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case bpf_map_type::BPF_MAP_TYPE_HASH:
		memory.destroy<hash_map_impl>(container_name.c_str());
		break;
	case bpf_map_type::BPF_MAP_TYPE_ARRAY:
		memory.destroy<array_map_impl>(container_name.c_str());
		break;
	case bpf_map_type::BPF_MAP_TYPE_RINGBUF:
		memory.destroy<ringbuf_map_impl>(container_name.c_str());
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY:
		memory.destroy<perf_event_array_map_impl>(
			container_name.c_str());
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_ARRAY:
		memory.destroy<per_cpu_array_map_impl>(container_name.c_str());
		break;
	case bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH:
		memory.destroy<per_cpu_hash_map_impl>(container_name.c_str());
		break;

	default:
		assert(false && "Unsupported map type");
	}
	map_impl_ptr = nullptr;
	return;
}

} // namespace bpftime
