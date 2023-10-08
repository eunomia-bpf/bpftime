#ifndef _HASHMAP_HPP
#define _HASHMAP_HPP
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <algorithm>
#include <bpf_map/map_common_def.hpp>
namespace bpftime
{

using namespace boost::interprocess;

struct BytesVecCompFunctor {
	bool operator()(const bytes_vec &a, const bytes_vec &b) const
	{
		if (a.size() != b.size())
			return a.size() < b.size();
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] == b[i])
				continue;
			return a[i] < b[i];
		}
		return false;
	}
};

// implementation of hash map
class hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, bytes_vec>;
	using bi_map_allocator =
		allocator<bi_map_value_ty,
			  managed_shared_memory::segment_manager>;
	boost::interprocess::map<bytes_vec, bytes_vec, BytesVecCompFunctor,
				 bi_map_allocator>
		map_impl;
	uint32_t _key_size;
	uint32_t _value_size;

	bytes_vec key_vec;
	bytes_vec value_vec;

    public:
    const static bool should_lock = true;
	hash_map_impl(managed_shared_memory &memory, uint32_t key_size,
		      uint32_t value_size);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int bpf_map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime
#endif
