#include "catch2/catch_message.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include "bpftime_config.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_handler.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "catch2/internal/catch_run_context.hpp"
#include <algorithm>
#include <random>
#include <map>

using namespace boost::interprocess;
using namespace bpftime;

static const char *SHM_NAME = "BPFTIME_TEST_ENTERNAL_MAP_SHM";
static const char *HANDLER_NAME = "my_handler";
static const int test_map_type = 2000;

static std::map<uint32_t, uint64_t> cpp_map;

static void *elem_lookup_static(int id, const void *key, bool from_syscall)
{
	auto k = *(uint32_t *)key;
	auto it = cpp_map.find(k);
	if (it == cpp_map.end())
		return nullptr;
	return &it->second;
}

static long elem_update_static(int id, const void *key, const void *value,
			       uint64_t flags, bool from_syscall)
{
	auto k = *(uint32_t *)key;
	auto v = *(uint32_t *)value;
	cpp_map[k] = v;
	return 0;
}

TEST_CASE("Test basic operations of external hash map ops")
{
	struct bpftime::shm_remove remover(SHM_NAME);
	// The side that creates the mapping
	managed_shared_memory segment(create_only, SHM_NAME, 1 << 20);
	auto manager =
		segment.construct<handler_manager>(HANDLER_NAME)(segment);
	auto &manager_ref = *manager;

	// update the ops to use the external map
	bpftime::agent_config agent_config;
	agent_config.allow_non_buildin_map_types = true;
	bpftime_set_agent_config(agent_config);

	cpp_map.clear();

	bpftime_map_ops map_ops{
		.elem_lookup = elem_lookup_static,
		.elem_update = elem_update_static,
	};

	int res = manager_ref.set_handler(1,
					  bpf_map_handler(1, test_map_type, 4,
							  8, 1024, 0, "hash1",
							  segment),
					  segment);
	REQUIRE(res == 1);

	REQUIRE(bpftime_register_map_ops(test_map_type, &map_ops) == 0);
	// test insert
	auto &map_handler = std::get<bpf_map_handler>(manager_ref[1]);
	for (uint64_t i = 0; i < 100; i++) {
		uint32_t key = i;
		uint64_t value = i;
		map_handler.map_update_elem(&key, &value, 0);
	}

	// test lookup
	for (uint64_t i = 0; i < 100; i++) {
		uint32_t key = i;
		auto val = *(uint64_t *)(map_handler.map_lookup_elem(&key));
		REQUIRE(val == i);
		spdlog::debug("val for {} = {:x}", i, val);
	}
}
