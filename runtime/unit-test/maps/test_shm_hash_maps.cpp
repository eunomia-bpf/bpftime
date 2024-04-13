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
	managed_shared_memory segment(create_only, SHM_NAME, 1 << 20);
	auto manager =
		segment.construct<handler_manager>(HANDLER_NAME)(segment);
	auto &manager_ref = *manager;

	manager_ref.set_handler(1,
				bpf_map_handler(1, BPF_MAP_TYPE_HASH, 4, 8, 1024,
						0, "hash1", segment),
				segment);
	manager_ref.set_handler(2,
				bpf_map_handler(2, BPF_MAP_TYPE_HASH, 4, 8, 1024,
						0, "hash2", segment),
				segment);
	manager_ref.set_handler(3,
				bpf_map_handler(3, BPF_MAP_TYPE_ARRAY, 4, 8, 1024,
						0, "array1", segment),
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
