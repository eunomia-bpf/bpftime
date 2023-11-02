#include "bpftool/libbpf/src/libbpf.h"
#include <bpf/libbpf.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
namespace bpftime
{

using mutex_ptr = boost::interprocess::managed_unique_ptr<
	boost::interprocess::interprocess_mutex,
	boost::interprocess::managed_shared_memory>::type;

// Here is an implementation of a perf event array that can output data from
// both userspace and kernel space It's corresponded with a normal perf event
// array in kernel, and it will be used by kernel ebpf programs But from the
// userspace, it will hold a user ringbuf (provided by libbpf). Once data was
// written, it will commit the data to the ringbuf. A kernel program will be
// attached to an intervally triggered event (e.g a timer perf event). This
// program will examine if data was available in the user ringbuf, and writes
// the data into the corresponding kernel perf event array.
class perf_event_array_kernel_user_impl {
	user_ring_buffer *user_rb = nullptr;
	uint32_t dummy = 0xffffffff;
	void init_user_ringbuf();
	uint32_t max_ent;
	int user_rb_id;
	int user_rb_fd;
	mutex_ptr reserve_mutex;

    public:
	const static bool should_lock = false;
	perf_event_array_kernel_user_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t value_size, uint32_t max_entries,
		int user_rb_id);
	virtual ~perf_event_array_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void ensure_init_user_ringbuf()
	{
		if (!user_rb)
			init_user_ringbuf();
	}
	int output_data(const void *buf, size_t size);
};
} // namespace bpftime
