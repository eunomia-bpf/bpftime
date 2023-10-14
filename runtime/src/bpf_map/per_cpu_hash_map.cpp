#include "bpf_map/map_common_def.hpp"
#include "spdlog/spdlog.h"
#include <bpf_map/per_cpu_hash_map.hpp>
#include <unistd.h>
namespace bpftime
{
per_cpu_hash_map_impl::per_cpu_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size)
	: impl(memory.get_segment_manager())
{
	int num_cpu = sysconf(_SC_NPROCESSORS_ONLN);
	for (int i = 0; i < num_cpu; i++) {
		impl.emplace_back(memory, key_size, value_size);
	}
}

void *per_cpu_hash_map_impl::elem_lookup(const void *key)
{
	return ensure_on_current_cpu<void *>([&](int cpu) {
		spdlog::debug("Run per cpu hash lookup at cpu {}", cpu);
		return impl[cpu].elem_lookup(key);
	});
}

long per_cpu_hash_map_impl::elem_update(const void *key, const void *value,
					uint64_t flags)
{
	return ensure_on_current_cpu<long>([&](int cpu) {
		spdlog::debug("Run per cpu hash update at cpu {}", cpu);
		return impl[cpu].elem_update(key, value, flags);
	});
}

long per_cpu_hash_map_impl::elem_delete(const void *key)
{
	return ensure_on_current_cpu<long>([&](int cpu) {
		spdlog::debug("Run per cpu hash delete at cpu {}", cpu);
		return impl[cpu].elem_delete(key);
	});
}

int per_cpu_hash_map_impl::bpf_map_get_next_key(const void *key, void *next_key)
{
	return ensure_on_current_cpu<int>([&](int cpu) {
		spdlog::debug("Run per cpu hash get next key at cpu {}", cpu);
		return impl[cpu].bpf_map_get_next_key(key, next_key);
	});
}

} // namespace bpftime
