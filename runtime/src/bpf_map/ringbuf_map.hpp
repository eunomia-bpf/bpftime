#ifndef _RINGBUF_MAP_HPP
#define _RINGBUF_MAP_HPP
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <cstddef>
namespace bpftime
{
using sharable_mutex_ptr = boost::interprocess::managed_unique_ptr<
	boost::interprocess::interprocess_sharable_mutex,
	boost::interprocess::managed_shared_memory>::type;
// implementation of ringbuf map
class ringbuf_map_impl {
	using vec_allocator = boost::interprocess::allocator<
		char,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using buf_vec = boost::interprocess::vector<char, vec_allocator>;
	using buf_vec_unique_ptr = boost::interprocess::managed_unique_ptr<
		buf_vec, boost::interprocess::managed_shared_memory>::type;
	uint32_t max_ent;
	uint32_t mask() const
	{
		return max_ent - 1;
	}
	boost::interprocess::offset_ptr<unsigned long> consumer_pos;
	boost::interprocess::offset_ptr<unsigned long> producer_pos;
	boost::interprocess::offset_ptr<uint8_t> data;
	// Guard for reserving memory
	mutable sharable_mutex_ptr reserve_mutex;
	// raw buffer
	buf_vec_unique_ptr raw_buffer;

    public:
	const static bool should_lock = false;
	ringbuf_map_impl(uint32_t max_ent,
			 boost::interprocess::managed_shared_memory &memory);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int bpf_map_get_next_key(const void *key, void *next_key);

	bool has_data() const;
	void *reserve(size_t size, int self_fd);
	void submit(const void *sample, bool discard);
};

} // namespace bpftime
#endif
