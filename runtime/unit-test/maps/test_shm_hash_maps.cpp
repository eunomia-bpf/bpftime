/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include "bpftime_handler.hpp"
#include <array>
#include <boost/interprocess/creation_tags.hpp>
#include <cstdint>
#include <linux/bpf.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *SHM_NAME = "my_shm_maps_test";
static const char *HANDLER_NAME = "my_handler";

using namespace boost::interprocess;
using namespace bpftime;

static void test_insert_map(int fd, bpftime::handler_manager &manager_ref,
			    managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint32_t key = i;
		uint64_t value = (((uint64_t)key) << 32) | 0xffffffff;
		map_handler.map_update_elem(&key, &value, 0);
	}
}

static void test_lookup_map(int fd, bpftime::handler_manager &manager_ref,
			    managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint32_t key = i;
		auto val = *(uint64_t *)(map_handler.map_lookup_elem(&key));
		REQUIRE(val == ((((uint64_t)key) << 32) | 0xffffffff));
		spdlog::debug("val for {} = {:x}", i, val);
	}
}

static void test_delete_map(int fd, bpftime::handler_manager &manager_ref,
			    managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	for (int i = 0; i < 100; i++) {
		uint64_t key = i;
		map_handler.map_delete_elem(&key);
	}
}

static void test_get_next_element(int fd, bpftime::handler_manager &manager_ref,
				  managed_shared_memory &segment)
{
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[fd]);
	uint32_t key = 0;
	uint32_t next_key = 0;
	while (map_handler.bpf_map_get_next_key(&key, &next_key) == 0) {
		spdlog::debug("key = {}, next_key = {}", key, next_key);
		key = next_key;
	}
	spdlog::debug("key = {}, next_key = {}", key, next_key);
}

static void handle_sub_process()
{
	spdlog::info("Subprocess entered");
	managed_shared_memory segment(open_only, SHM_NAME);
	auto manager = segment.find<handler_manager>(HANDLER_NAME).first;
	auto &manager_ref = *manager;
	manager_ref.clear_id_at(2, segment);
	test_lookup_map(1, manager_ref, segment);
	test_lookup_map(3, manager_ref, segment);
	test_get_next_element(1, manager_ref, segment);
	test_get_next_element(3, manager_ref, segment);
	test_delete_map(1, manager_ref, segment);
	test_delete_map(3, manager_ref, segment);
	manager->clear_all(segment);
	spdlog::info("Print maps value finished");
	segment.destroy_ptr(manager);
	spdlog::info("Subprocess exited");
	_exit(0);
}

TEST_CASE("Test shm hash maps with sub process")
{
	struct shm_remove remover(SHM_NAME);

	// The side that creates the mapping
	// Needs enough room for the handler table + multiple map instances.
	managed_shared_memory segment(create_only, SHM_NAME, 8 << 20);
	// Only a handful of fds are used in this test; keep max_fd_count small
	// so the handler table doesn't dominate the shared memory segment.
	const size_t test_max_fd_count = MIN_MAX_FD_COUNT;
	auto manager = segment.construct<handler_manager>(HANDLER_NAME)(
		segment, test_max_fd_count);
	auto &manager_ref = *manager;

	manager_ref.set_handler(1,
				bpf_map_handler(1, BPF_MAP_TYPE_HASH, 4, 8,
						1024, 0, "hash1", segment),
				segment);
	manager_ref.set_handler(2,
				bpf_map_handler(2, BPF_MAP_TYPE_HASH, 4, 8,
						1024, 0, "hash2", segment),
				segment);
	manager_ref.set_handler(3,
				bpf_map_handler(3, BPF_MAP_TYPE_ARRAY, 4, 8,
						1024, 0, "array1", segment),
				segment);

	// test insert
	test_insert_map(1, manager_ref, segment);
	test_insert_map(3, manager_ref, segment);
	test_get_next_element(1, manager_ref, segment);
	test_get_next_element(3, manager_ref, segment);
	// test lookup
	test_lookup_map(1, manager_ref, segment);
	test_lookup_map(3, manager_ref, segment);
	spdlog::info("Starting subprocess");
	int pid = fork();
	if (pid == 0) {
		handle_sub_process();
	} else {
		int status;
		int ret = waitpid(pid, &status, 0);
		REQUIRE(ret != -1);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
		REQUIRE(segment.find<handler_manager>(HANDLER_NAME).first ==
			nullptr);
	}
}

TEST_CASE("Hash maps with colliding public names remain independent",
	  "[maps][map_name]")
{
	static const char *collision_shm_name =
		"bpftime_map_name_collision_test";
	struct shm_remove remover(collision_shm_name);
	managed_shared_memory segment(create_only, collision_shm_name, 8 << 20);
	auto *manager = segment.construct<handler_manager>(
		HANDLER_NAME)(segment, MIN_MAX_FD_COUNT);
	const auto named_objects_before_maps = segment.get_num_named_objects();
	const auto free_memory_before_maps = segment.get_free_memory();

	bpf_map_attr attr{
		.type = BPF_MAP_TYPE_HASH,
		.key_size = sizeof(uint32_t),
		.value_size = sizeof(uint64_t),
		.max_ents = 16,
	};
	/* The first pair models two kernel-truncated names; the remaining pairs
	 * cover identical short names and anonymous public names.
	 */
	constexpr std::array<const char *, 6> names = {
		"libc_malloc_cal", "libc_malloc_cal", "same", "same", "", ""
	};
	std::array<uint64_t, names.size()> expected_values{};
	const uint32_t key = 1;

	for (std::size_t i = 0; i < names.size(); i++) {
		const int fd = static_cast<int>(i + 1);
		REQUIRE(manager->set_handler(fd,
					     bpf_map_handler(fd, names[i],
							     segment, attr),
					     segment) == fd);
		REQUIRE(segment.get_num_named_objects() ==
			named_objects_before_maps);

		auto &map = std::get<bpf_map_handler>((*manager)[fd]);
		REQUIRE(std::string(map.name.c_str()) == names[i]);
		expected_values[i] = 100 + i;
		REQUIRE(map.map_update_elem(&key, &expected_values[i],
					    BPF_ANY) == 0);
	}

	for (std::size_t i = 0; i < names.size(); i++) {
		auto &map = std::get<bpf_map_handler>(
			(*manager)[static_cast<int>(i + 1)]);
		auto *value = static_cast<const uint64_t *>(
			map.map_lookup_elem(&key));
		REQUIRE(value != nullptr);
		REQUIRE(*value == expected_values[i]);
	}

	constexpr int source_fd = 1;
	constexpr int duplicate_fd = 7;
	auto &source = std::get<bpf_map_handler>((*manager)[source_fd]);
	bpf_map_handler duplicate(duplicate_fd, source.name.c_str(), segment,
				  source.attr);
	duplicate.share_map_impl_from(source);
	source.inc_map_refcount();
	REQUIRE(manager->set_handler(duplicate_fd, std::move(duplicate),
				     segment) == duplicate_fd);

	manager->clear_id_at(source_fd, segment);
	auto &shared_map = std::get<bpf_map_handler>((*manager)[duplicate_fd]);
	auto *shared_value =
		static_cast<const uint64_t *>(shared_map.map_lookup_elem(&key));
	REQUIRE(shared_value != nullptr);
	REQUIRE(*shared_value == expected_values[0]);

	manager->clear_id_at(duplicate_fd, segment);
	auto &independent_peer = std::get<bpf_map_handler>((*manager)[2]);
	auto *peer_value = static_cast<const uint64_t *>(
		independent_peer.map_lookup_elem(&key));
	REQUIRE(peer_value != nullptr);
	REQUIRE(*peer_value == expected_values[1]);

	manager->clear_all(segment);
	REQUIRE(segment.get_num_named_objects() == named_objects_before_maps);
	REQUIRE(segment.get_free_memory() == free_memory_before_maps);
	segment.destroy_ptr(manager);
	shared_memory_object::remove(collision_shm_name);
}
