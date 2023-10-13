#include <bpf_map/hash_map.hpp>
#include <algorithm>
#include <functional>
#include <unistd.h>
namespace bpftime
{
hash_map_impl::hash_map_impl(managed_shared_memory &memory, uint32_t key_size,
			     uint32_t value_size)
	: map_impl(10, bytes_vec_hasher(), std::equal_to<bytes_vec>(),
		   bi_map_allocator(memory.get_segment_manager())),
	  _key_size(key_size), _value_size(value_size),
	  key_vec(key_size, memory.get_segment_manager()),
	  value_vec(value_size, memory.get_segment_manager())
{
}
void *hash_map_impl::elem_lookup(const void *key)
{
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	bytes_vec key_vec = this->key_vec;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	if (auto itr = map_impl.find(key_vec); itr != map_impl.end()) {
		return &itr->second[0];
	} else {
		errno = ENOENT;
		return nullptr;
	}
}

long hash_map_impl::elem_update(const void *key, const void *value,
				uint64_t flags)
{
	bytes_vec key_vec = this->key_vec;
	bytes_vec value_vec = this->value_vec;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	value_vec.assign((uint8_t *)value, (uint8_t *)value + _value_size);
	if (auto itr = map_impl.find(key_vec); itr != map_impl.end()) {
		itr->second = value_vec;
	} else {
		map_impl.insert(bi_map_value_ty(key_vec, value_vec));
	}
	return 0;
}

long hash_map_impl::elem_delete(const void *key)
{
	bytes_vec key_vec = this->key_vec;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	map_impl.erase(key_vec);
	return 0;
}

int hash_map_impl::bpf_map_get_next_key(const void *key, void *next_key)
{
	if (key == nullptr) {
		// nullptr means the first key
		auto itr = map_impl.begin();
		if (itr == map_impl.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	// No need to be allocated at shm. Allocate as a local variable to make
	// it thread safe, since we use sharable lock
	bytes_vec key_vec = this->key_vec;
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		// not found, should be refer to the first key
		return bpf_map_get_next_key(nullptr, next_key);
	}
	itr++;
	if (itr == map_impl.end()) {
		// If *key* is the last element, returns -1 and *errno*
		// is set to **ENOENT**.
		errno = ENOENT;
		return -1;
	}
	std::copy(itr->first.begin(), itr->first.end(), (uint8_t *)next_key);
	return 0;
}

} // namespace bpftime
