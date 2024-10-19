#include "catch2/catch_message.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
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

using namespace boost::interprocess;
using namespace bpftime;

static const char *SHM_NAME = "BPFTIME_PER_CPU_HASH_SHM";

TEST_CASE("Test basic operations of hash map")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	uint32_t ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	std::mt19937 gen;
	gen.seed(Catch::rngSeed());
	// Since this is a hash map, so we generate something randomly as keys
	std::uniform_int_distribution<uint32_t> rand(0, UINT32_MAX);
	std::vector<uint32_t> keys;
	for (int i = 0; i < 100; i++) {
		auto x = rand(gen);
		while (std::find(keys.begin(), keys.end(), x) != keys.end())
			x = rand(gen);
		keys.push_back(x);
	}

	SECTION("Test writing from helpers, and read from userspace")
	{
		per_cpu_hash_map_impl map(mem, 4, 8, 1 << 20);
		for (uint32_t j = 0; j < ncpu; j++) {
			ensure_on_certain_cpu<void>(j, [&]() {
				for (uint32_t i = 0; i < 100; i++) {
					uint64_t val =
						(((uint64_t)keys[i]) << 32) | j;
					INFO("Set index " << keys[i] << " cpu "
							  << j << " to "
							  << val);
					REQUIRE(map.elem_update(&keys[i], &val,
								0) == 0);
				}
			});
		}
		for (uint32_t i = 0; i < 100; i++) {
			uint64_t *p =
				(uint64_t *)map.elem_lookup_userspace(&keys[i]);
			for (uint32_t j = 0; j < ncpu; j++) {
				REQUIRE(p[j] ==
					((((uint64_t)keys[i]) << 32) | j));
			}
		}
	}

	SECTION("Test writing from userspace, and reading & updating from helpers")
	{
		per_cpu_hash_map_impl map(mem, 4, 8, 1 << 20);
		std::vector<uint64_t> buf(ncpu);
		for (uint32_t j = 0; j < ncpu; j++) {
			buf[j] = j;
		}
		uint32_t dummy = 0xabcdef;
		REQUIRE(map.elem_update_userspace(&dummy, buf.data(), 0) == 0);
		// Update from helpers
		for (uint32_t i = 0; i < ncpu; i++) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&dummy);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				value |= (value << 32);
				REQUIRE(map.elem_update(&dummy, &value, 0) ==
					0);
			});
		}
		// Read from helpers
		for (uint32_t i = ncpu - 1; ~i; i--) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&dummy);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				uint64_t a = value >> 32;
				uint64_t b = (value << 32) >> 32;
				REQUIRE(a == b);
			});
		}
	}
}
