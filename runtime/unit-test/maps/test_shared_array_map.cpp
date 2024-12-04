#include "bpf/bpf.h"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "catch2/catch_test_macros.hpp"
#include "handler/handler_manager.hpp"
#include <cerrno>
#include <cstdint>

static const char *SHM_NAME = "_bpftime_shared_array_map_test";

TEST_CASE("Test shared array map")
{
	struct bpftime::shm_remove remover(SHM_NAME);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	auto &shm = bpftime::shm_holder.global_shared_memory;

	auto &manager_ref = shm.get_manager_mut();
	auto &segment = shm.get_segment_manager();
	LIBBPF_OPTS(bpf_map_create_opts, opts);
	opts.map_flags = BPF_F_MMAPABLE;
	int fd = bpf_map_create((enum bpf_map_type)BPF_MAP_TYPE_ARRAY,
				"test_map", 4, 4, 1024, &opts);
	REQUIRE(fd > 0);
	struct bpf_map_info info;
	uint32_t len = sizeof(info);
	REQUIRE(bpf_map_get_info_by_fd(fd, &info, &len) == 0);

	manager_ref.set_handler(
		1,
		bpftime::bpf_map_handler(
			1, "test_map", segment,
			bpftime::bpf_map_attr{
				.type = (int)bpftime::bpf_map_type::
					BPF_MAP_TYPE_KERNEL_USER_ARRAY,
				.key_size = 4,
				.value_size = 4,
				.max_ents = 1024,
				.kernel_bpf_map_id = info.id }),
		segment);

	auto &map = std::get<bpftime::bpf_map_handler>(manager_ref[1]);
	int32_t key = 1023;
	int32_t value = 2;
	REQUIRE(map.map_update_elem(&key, &value, 0, true) == 0);
	// Test get next key
	key = 1;
	int32_t next_key;
	REQUIRE(map.bpf_map_get_next_key(&key, &next_key, true) == 0);
	REQUIRE(next_key == 2);
	key = 1023;
	REQUIRE((map.bpf_map_get_next_key(&key, &next_key, true) < 0 &&
		 errno == ENOENT));
	key = 1;
	REQUIRE(map.map_delete_elem(&key, true) == 0);
	auto ptr = map.map_lookup_elem(&key, true);
	REQUIRE(ptr != nullptr);
	REQUIRE(*(int32_t *)ptr == 0);
}
