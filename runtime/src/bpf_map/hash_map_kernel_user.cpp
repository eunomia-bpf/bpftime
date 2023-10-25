#include "spdlog/spdlog.h"
#include <bpf_map/hash_map_kernel_user.hpp>
#include <algorithm>
#include <functional>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

namespace bpftime
{
hash_map_kernel_user_impl::hash_map_kernel_user_impl(
	managed_shared_memory &memory, int km_id) : key_vec(1, memory.get_segment_manager()),
	  value_vec(1, memory.get_segment_manager())
{
	map_fd = bpf_map_get_fd_by_id(kernel_map_id);
	if (map_fd < 0) {
		spdlog::error("Failed to get fd for kernel map id {}",
			      kernel_map_id);
		return;
	}
	kernel_map_id = km_id;
	bpf_map_info info = {};
	unsigned int info_len = sizeof(info);
	int res = bpf_map_get_info_by_fd(map_fd, &info, &info_len);
	if (res < 0) {
		spdlog::error("Failed to get info for kernel map id {}",
			      kernel_map_id);
	}
	_key_size = info.key_size;
	_value_size = info.value_size;
	spdlog::debug("create kernel user hash map key size {}, value size {}",
		      _key_size, _value_size);
	key_vec.resize(_key_size);
	value_vec.resize(_value_size);
}

void *hash_map_kernel_user_impl::elem_lookup(const void *key)
{
	spdlog::trace("Peform elem lookup of hash map");
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	int res = bpf_map_lookup_elem(map_fd, key, value_vec.data());
	if (res < 0) {
		return nullptr;
	}
	return value_vec.data();
}

long hash_map_kernel_user_impl::elem_update(const void *key, const void *value,
					    uint64_t flags)
{
	spdlog::trace("Peform elem update of hash map");
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_update_elem(map_fd, key, value, flags);
}

long hash_map_kernel_user_impl::elem_delete(const void *key)
{
	spdlog::trace("Peform elem delete of hash map");
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_delete_elem(map_fd, key);
}

int hash_map_kernel_user_impl::map_get_next_key(const void *key,
						    void *next_key)
{
	spdlog::trace("Peform get next key of hash map");
	// Allocate as a local variable to make
	//  it thread safe, since we use sharable lock
	return bpf_map_get_next_key(map_fd, key, next_key);
}

} // namespace bpftime
