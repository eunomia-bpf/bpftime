#ifndef _ARRAY_MAP_HPP
#define _ARRAY_MAP_HPP
#include <bpf_map/maps.hpp>
namespace bpftime
{

// implementation of array map
class array_map_impl {
	bytes_vec data;
	uint32_t _value_size;
	uint32_t _max_entries;

    public:
	const static bool should_lock = true;
	array_map_impl(boost::interprocess::managed_shared_memory &memory,
		       uint32_t value_size, uint32_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int bpf_map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime
#endif
