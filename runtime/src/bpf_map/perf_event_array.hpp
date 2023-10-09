#ifndef _PERF_EVENT_ARRAY_HPP
#define _PERF_EVENT_ARRAY_HPP
#include <cstdint>
#include <boost/interprocess/managed_shared_memory.hpp>
namespace bpftime
{
class perf_event_array_impl {
    private:
    
    public:
	const static bool should_lock = true;
	perf_event_array_impl(boost::interprocess::managed_shared_memory &memory,
			      uint32_t key_size, uint32_t value_size);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int bpf_map_get_next_key(const void *key, void *next_key);
};
} // namespace bpftime
#endif
