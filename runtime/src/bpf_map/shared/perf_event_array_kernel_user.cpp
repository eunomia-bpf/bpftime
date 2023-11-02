#include "bpf/bpf.h"
#include "bpf/libbpf_common.h"
#include "bpftool/libbpf/src/libbpf.h"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <bpf_map/shared/perf_event_array_kernel_user.hpp>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <spdlog/spdlog.h>
namespace bpftime
{

int perf_event_array_kernel_user_impl::output_data(const void *buf, size_t size)
{
	ensure_init_user_ringbuf();
	void *mem = nullptr;
	{
		std::scoped_lock<boost::interprocess::interprocess_mutex> guard(
			*reserve_mutex);
		// Reserving is *NOT* thread safe
		mem = user_ring_buffer__reserve(user_rb, size);
	}
	if (!mem) {
		spdlog::error("Failed to reserve for user ringbuf: {}", errno);
		return errno;
	}
	memcpy(mem, buf, size);
	user_ring_buffer__submit(user_rb, mem);
	return 0;
}
void perf_event_array_kernel_user_impl::init_user_ringbuf()
{
	user_rb_fd = bpf_map_get_fd_by_id(user_rb_id);
	if (user_rb_fd < 0) {
		spdlog::error(
			"Failed to get user ringbuf fd from id {}, err={}",
			user_rb_id, user_rb_fd);
		return;
	}
	LIBBPF_OPTS(user_ring_buffer_opts, opts);

	user_rb = user_ring_buffer__new(user_rb_fd, &opts);
	if (!user_rb) {
		spdlog::error("Failed to create user ringbuf for fd {}, err={}",
			      user_rb_fd, errno);
		return;
	}
}
perf_event_array_kernel_user_impl::perf_event_array_kernel_user_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries, int user_rb_id)
	: max_ent(max_entries), user_rb_id(user_rb_id),
	  reserve_mutex(boost::interprocess::make_managed_unique_ptr(
		  memory.construct<boost::interprocess::interprocess_mutex>(
			  boost::interprocess::anonymous_instance)(),
		  memory))
{
	if (key_size != 4 || value_size != 4) {
		spdlog::error(
			"Key size and value size of perf_event_array must be 4");
		assert(false);
	}
	ensure_init_user_ringbuf();
}
perf_event_array_kernel_user_impl::~perf_event_array_kernel_user_impl()
{
	user_ring_buffer__free(user_rb);
}

void *perf_event_array_kernel_user_impl::elem_lookup(const void *key)
{
	ensure_init_user_ringbuf();
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return nullptr;
	}
	spdlog::info(
		"Looking up key {} from perf event array kernel user, which is useless",
		k);
	return &dummy;
}

long perf_event_array_kernel_user_impl::elem_update(const void *key,
						    const void *value,
						    uint64_t flags)
{
	ensure_init_user_ringbuf();
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	int32_t v = *(int32_t *)value;
	spdlog::info(
		"Updating key {}, value {} from perf event array kernel user, which is useless",
		k, v);
	return 0;
}

long perf_event_array_kernel_user_impl::elem_delete(const void *key)
{
	ensure_init_user_ringbuf();
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	spdlog::info(
		"Deleting key {} from perf event array kernel user, which is useless",
		k);
	return 0;
}

int perf_event_array_kernel_user_impl::map_get_next_key(const void *key,
							void *next_key)
{
	ensure_init_user_ringbuf();
	int32_t *out = (int32_t *)next_key;
	if (key == nullptr) {
		*out = 0;
		return 0;
	}
	int32_t k = *(int32_t *)key;
	// The last key
	if ((size_t)(k + 1) == max_ent) {
		errno = ENOENT;
		return -1;
	}
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	*out = k + 1;
	return 0;
}
} // namespace bpftime
