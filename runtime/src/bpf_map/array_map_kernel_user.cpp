#include <bpf_map/array_map_kernel_user.hpp>
#include <cerrno>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

namespace bpftime
{

void *array_map_kernel_user_impl::get_raw_data() const
{
	return (void *)data.data();
}

void array_map_kernel_user_impl::init_map_fd()
{
	map_fd = bpf_map_get_fd_by_id(kernel_map_id);
	if (map_fd < 0) {
		spdlog::error("Failed to get fd for kernel map id {}",
			      kernel_map_id);
		return;
	}
	bpf_map_info info = {};
	unsigned int info_len = sizeof(info);
	int res = bpf_map_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get info for kernel map id {}",
			      kernel_map_id);
	}
	_value_size = info.value_size;
	_max_entries = info.max_entries;
	spdlog::debug(
		"create kernel user array map value size {}, max entries {}",
		_value_size, _max_entries);
	data.resize(_value_size * _max_entries);
}

array_map_kernel_user_impl::array_map_kernel_user_impl(
	boost::interprocess::managed_shared_memory &memory, int km_id)
	: data(1, memory.get_segment_manager()), kernel_map_id(km_id)
{
}

void *array_map_kernel_user_impl::elem_lookup(const void *key)
{
	if (map_fd < 0) {
		init_map_fd();
	}
	auto key_val = *(uint32_t *)key;
	if (key_val >= _max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	return &data[key_val * _value_size];
}

long array_map_kernel_user_impl::elem_update(const void *key, const void *value,
					     uint64_t flags)
{
	if (map_fd < 0) {
		init_map_fd();
	}
	auto key_val = *(uint32_t *)key;
	if (key_val >= _max_entries) {
		errno = ENOENT;
		return -1;
	}
	std::copy((uint8_t *)value, (uint8_t *)value + _value_size,
		  &data[key_val * _value_size]);
	return 0;
}

long array_map_kernel_user_impl::elem_delete(const void *key)
{
	if (map_fd < 0) {
		init_map_fd();
	}
	auto key_val = *(uint32_t *)key;
	if (key_val >= _max_entries) {
		errno = ENOENT;
		return -1;
	}
	std::fill(&data[key_val * _value_size],
		  &data[key_val * _value_size] + _value_size, 0);
	return 0;
}

int array_map_kernel_user_impl::map_get_next_key(const void *key,
						 void *next_key)
{
	// Not found
	if (key == nullptr || *(uint32_t *)key >= _max_entries) {
		*(uint32_t *)next_key = 0;
		return 0;
	}
	uint32_t deref_key = *(uint32_t *)key;
	// Last element
	if (deref_key == _max_entries - 1) {
		errno = ENOENT;
		return -1;
	}
	auto key_val = *(uint32_t *)key;
	*(uint32_t *)next_key = key_val + 1;
	return 0;
}

array_map_kernel_user_impl::~array_map_kernel_user_impl() {
	if (map_fd >= 0) {
		close(map_fd);
	}
}

} // namespace bpftime
