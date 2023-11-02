#include "catch2/catch_message.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_array_map.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "../common_def.hpp"

using namespace boost::interprocess;
using namespace bpftime;

static const char *SHM_NAME = "_PER_CPU_ARRAY_SHM";

TEST_CASE("Test basic operations of array map")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);
	uint32_t ncpu = sysconf(_SC_NPROCESSORS_ONLN);

	SECTION("Test writing from helpers, and read from userspace")
	{
		per_cpu_array_map_impl map(mem, 8, 10);
		for (uint32_t j = 0; j < ncpu; j++) {
			ensure_on_certain_cpu<void>(j, [&]() {
				for (uint32_t i = 0; i < 10; i++) {
					uint64_t val =
						(((uint64_t)i) << 32) | j;
					INFO("Set index " << i << " cpu " << j
							  << " to " << val);
					REQUIRE(map.elem_update(&i, &val, 0) ==
						0);
				}
			});
		}
		for (uint32_t i = 0; i < 10; i++) {
			uint64_t *p = (uint64_t *)map.elem_lookup_userspace(&i);
			for (uint32_t j = 0; j < ncpu; j++) {
				REQUIRE(p[j] == ((((uint64_t)i) << 32) | j));
			}
		}
	}

	SECTION("Test writing from userspace, and reading & updating from helpers")
	{
		per_cpu_array_map_impl map(mem, 8, 1);
		std::vector<uint64_t> buf(ncpu);
		for (uint32_t j = 0; j < ncpu; j++) {
			buf[j] = j;
		}
		uint32_t zero = 0;
		REQUIRE(map.elem_update_userspace(&zero, buf.data(), 0) == 0);
		// Update from helpers
		for (uint32_t i = 0; i < ncpu; i++) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&zero);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				value |= (value << 32);
				REQUIRE(map.elem_update(&zero, &value, 0) == 0);
			});
		}
		// Read from helpers
		for (uint32_t i = ncpu - 1; ~i; i--) {
			ensure_on_certain_cpu<void>(i, [&]() {
				auto valuep =
					(uint64_t *)map.elem_lookup(&zero);
				REQUIRE(valuep != nullptr);
				auto value = *valuep;
				uint64_t a = value >> 32;
				uint64_t b = (value << 32) >> 32;
				REQUIRE(a == b);
			});
		}
	}
}
