#include "bpf_map/map_common_def.hpp"
#include "spdlog/fmt/bin_to_hex.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <bpf_map/per_cpu_hash_map.hpp>
#include <unistd.h>
namespace bpftime
{
per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size)
	: per_cpu_hash_map_impl(memory, key_size, value_size,
				sysconf(_SC_NPROCESSORS_ONLN))
{
}

per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, int ncpu)
	: impl(memory.get_segment_manager()), key_size(key_size),
	  value_size(value_size), ncpu(ncpu),
	  key_template(key_size, memory.get_segment_manager()),
	  value_template(value_size * ncpu, memory.get_segment_manager()),
	  single_value_template(value_size, memory.get_segment_manager())
{
	spdlog::debug(
		"Initializing per cpu hash, key size {}, value size {}, ncpu {}",
		key_size, value_size, ncpu);
}

void *per_cpu_hash_map_impl::elem_lookup(const void *key)
{
	return ensure_on_current_cpu<void *>([&](int cpu) -> void * {
		spdlog::debug("Run per cpu hash lookup at cpu {}", cpu);
		if (key == nullptr) {
			errno = ENOENT;
			return nullptr;
		}
		bytes_vec key_vec = this->key_template;
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		if (auto itr = impl.find(key_vec); itr != impl.end()) {
			spdlog::trace("Exit elem lookup of hash map");
			return &itr->second[value_size * cpu];
		} else {
			spdlog::trace("Exit elem lookup of hash map");
			errno = ENOENT;
			return nullptr;
		}
	});
}

long per_cpu_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	spdlog::debug("Per cpu update, key {}, value {}",(const char*)key,*(long*)value);
	return ensure_on_current_cpu<long>([&](int cpu) -> long {
		spdlog::debug("Run per cpu hash update at cpu {}", cpu);
		bytes_vec key_vec = this->key_template;
		bytes_vec value_vec = this->single_value_template;
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		value_vec.assign((uint8_t *)value,
				 (uint8_t *)value + value_size);
		if (auto itr = impl.find(key_vec); itr != impl.end()) {
			std::copy(value_vec.begin(), value_vec.end(),
				  itr->second.begin() + cpu * value_size);
		} else {
			bytes_vec full_value_vec = this->value_template;
			std::copy(value_vec.begin(), value_vec.end(),
				  full_value_vec.begin() + cpu * value_size);

			impl.insert(bi_map_value_ty(key_vec, full_value_vec));
		}

		return 0;
	});
}

long per_cpu_hash_map_impl::elem_delete(const void *key)
{
	return ensure_on_current_cpu<long>([&](int cpu) -> long {
		spdlog::debug("Run per cpu hash delete at cpu {}", cpu);
		bytes_vec key_vec = this->key_template;
		key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
		if (auto itr = impl.find(key_vec); itr != impl.end()) {
			std::fill(itr->second.begin(),
				  itr->second.begin() + cpu * value_size, 0);
		}
		return 0;
	});
}

int per_cpu_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr) {
		// nullptr means the first key
		auto itr = impl.begin();
		if (itr == impl.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	// No need to be allocated at shm. Allocate as a local variable to make
	// it thread safe, since we use sharable lock
	bytes_vec key_vec = this->key_template;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);

	auto itr = impl.find(key_vec);
	if (itr == impl.end()) {
		// not found, should be refer to the first key
		return map_get_next_key(nullptr, next_key);
	}
	itr++;
	if (itr == impl.end()) {
		// If *key* is the last element, returns -1 and *errno*
		// is set to **ENOENT**.
		errno = ENOENT;
		return -1;
	}
	std::copy(itr->first.begin(), itr->first.end(), (uint8_t *)next_key);
	return 0;
}
void *per_cpu_hash_map_impl::elem_lookup_userspace(const void *key)
{
	if (key == nullptr) {
		errno = ENOENT;
		return nullptr;
	}
	bytes_vec key_vec = this->key_template;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	if (auto itr = impl.find(key_vec); itr != impl.end()) {
		spdlog::trace("Exit elem lookup of hash map: {}",spdlog::to_hex(itr->second.begin(),itr->second.end()));
		return &itr->second[0];
	} else {
		spdlog::trace("Exit elem lookup of hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long per_cpu_hash_map_impl::elem_update_userspace(const void *key,
						  const void *value,
						  uint64_t flags)
{
	bytes_vec key_vec = this->key_template;
	bytes_vec value_vec = this->value_template;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	value_vec.assign((uint8_t *)value,
			 (uint8_t *)value + value_size * ncpu);

	if (auto itr = impl.find(key_vec); itr != impl.end()) {
		itr->second = value_vec;
	} else {
		impl.insert(bi_map_value_ty(key_vec, value_vec));
	}
	return 0;
}
long per_cpu_hash_map_impl::elem_delete_userspace(const void *key)
{
	bytes_vec key_vec = this->key_template;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + key_size);
	impl.erase(key_vec);
	return 0;
}
} // namespace bpftime
