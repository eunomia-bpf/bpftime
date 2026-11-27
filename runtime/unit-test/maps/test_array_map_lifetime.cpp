// SPDX-License-Identifier: MIT
#include "catch2/catch_test_macros.hpp"
#include "bpftime_shm_internal.hpp"
#include "../common_def.hpp"

using namespace bpftime;

TEST_CASE("Array backing survives fd close and reuse", "[array_lifetime]")
{
	::shm_remove remover("_ARRAY_LIFETIME_SHM");
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	constexpr int fd = 1201;
	const bpf_map_attr attr{ .type = int(bpf_map_type::BPF_MAP_TYPE_ARRAY),
				 .key_size = 4,
				 .value_size = 8,
				 .max_ents = 2 };
	REQUIRE(bpftime_maps_create(fd, "retained", attr) == fd);
	auto &shm = shm_holder.global_shared_memory;
	auto retained = shm.retain_array_map(fd);
	REQUIRE(retained);
	uint32_t key = 0;
	uint64_t value = 123;
	REQUIRE(retained->elem_update(&key, &value, 0) == 0);
	bpftime_close(fd);
	REQUIRE(bpftime_maps_create(fd, "replacement", attr) == fd);
	REQUIRE(*static_cast<uint64_t *>(retained->elem_lookup(&key)) == 123);
	*static_cast<uint64_t *>(retained->elem_lookup(&key)) = 456;
	REQUIRE(*static_cast<const uint64_t *>(
			bpftime_map_lookup_elem(fd, &key)) == 0);
	retained.reset();
	bpftime_close(fd);
	bpftime_destroy_global_shm();
}
