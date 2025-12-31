/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include "bpftime_handler.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <cstdint>
#include <linux/bpf.h>
#include <memory>
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
	std::unique_ptr<managed_shared_memory> segment;
	REQUIRE_NOTHROW(segment = std::make_unique<managed_shared_memory>(
				create_only, SHM_NAME, 8 << 20));
	auto &segment_ref = *segment;
	// Only a handful of fds are used in this test; keep max_fd_count small
	// so the handler table doesn't dominate the (intentionally small) shm.
	const size_t test_max_fd_count = MIN_MAX_FD_COUNT;
	handler_manager *manager = nullptr;
	REQUIRE_NOTHROW(
		manager = segment_ref.construct<handler_manager>(HANDLER_NAME)(
			segment_ref, test_max_fd_count));
	auto &manager_ref = *manager;

	REQUIRE_NOTHROW(manager_ref.set_handler(
		1,
		bpf_map_handler(1, BPF_MAP_TYPE_HASH, 4, 8, 1024, 0, "hash1",
				segment_ref),
		segment_ref));
	REQUIRE_NOTHROW(manager_ref.set_handler(
		2,
		bpf_map_handler(2, BPF_MAP_TYPE_HASH, 4, 8, 1024, 0, "hash2",
				segment_ref),
		segment_ref));
	REQUIRE_NOTHROW(manager_ref.set_handler(
		3,
		bpf_map_handler(3, BPF_MAP_TYPE_ARRAY, 4, 8, 1024, 0, "array1",
				segment_ref),
		segment_ref));

	// test insert
	REQUIRE_NOTHROW(test_insert_map(1, manager_ref, segment_ref));
	REQUIRE_NOTHROW(test_insert_map(3, manager_ref, segment_ref));
	REQUIRE_NOTHROW(test_get_next_element(1, manager_ref, segment_ref));
	REQUIRE_NOTHROW(test_get_next_element(3, manager_ref, segment_ref));
	// test lookup
	REQUIRE_NOTHROW(test_lookup_map(1, manager_ref, segment_ref));
	REQUIRE_NOTHROW(test_lookup_map(3, manager_ref, segment_ref));
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
		REQUIRE(segment_ref.find<handler_manager>(HANDLER_NAME).first ==
			nullptr);
	}
}
