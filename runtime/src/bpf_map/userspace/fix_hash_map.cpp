/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <bpf_map/userspace/fix_hash_map.hpp>
#include <cerrno>
#include <linux/bpf.h>

static bool is_good_update_flag(uint64_t flags)
{
	return flags == BPF_ANY || flags == BPF_EXIST || flags == BPF_NOEXIST;
}

namespace bpftime
{

fix_size_hash_map_impl::fix_size_hash_map_impl(managed_shared_memory &memory,
					       size_t num_buckets,
					       size_t key_size,
					       size_t value_size)
	: map_impl(memory, num_buckets, key_size, value_size),
	  _value_size(value_size)
{
}

void *fix_size_hash_map_impl::elem_lookup(const void *key)
{
	return map_impl.elem_lookup(key);
}

long fix_size_hash_map_impl::elem_update(const void *key, const void *value,
					 uint64_t flags, bool from_syscall)
{
	if (!is_good_update_flag(flags)) {
		errno = EINVAL;
		return -1;
	}
	return map_impl.upsert(key, value, flags, from_syscall, 0, _value_size);
}

long fix_size_hash_map_impl::elem_delete(const void *key, bool from_syscall)
{
	return map_impl.erase(key, nullptr, from_syscall);
}

int fix_size_hash_map_impl::lookup_and_delete(const void *key, void *value_out,
					      bool from_syscall)
{
	if (value_out == nullptr) {
		errno = EINVAL;
		return -1;
	}
	return (int)map_impl.erase(key, value_out, from_syscall);
}

int fix_size_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	return map_impl.map_get_next_key(key, next_key);
}

} // namespace bpftime
