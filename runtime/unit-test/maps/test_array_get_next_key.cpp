/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * Kernel `array_map_get_next_key` contract for the two sparse array
 * backends (prog array, perf-event array): the walk is dense over the
 * index range, an absent or out-of-range key restarts at index 0, and only
 * the last index reports -ENOENT. Reported in #782.
 */
#include "catch2/catch_test_macros.hpp"

#include "bpftime_shm.hpp"
#include "../common_def.hpp"
#include <cerrno>
#include <cstdint>
#include <unistd.h>

#if __linux__
#include <linux/bpf.h>
#include <linux/filter.h>

using namespace bpftime;

namespace
{
constexpr int prog_array_fd = 1201;
constexpr int perf_array_fd = 1202;
constexpr int target_prog_fd = 1203;

void create_map_or_skip(int fd, const char *name, int type, uint32_t max_ents)
{
	REQUIRE(bpftime_maps_create(fd, name,
				    bpf_map_attr{ .type = type,
						  .key_size = 4,
						  .value_size = 4,
						  .max_ents = max_ents }) >= 0);
}

// The three transitions the kernel defines, for one backend.
void check_array_iteration(int fd)
{
	uint32_t next = 0xffffffffu;
	errno = 0;

	// A null key starts at the first index; -ENOENT must not leak.
	REQUIRE(bpftime_map_get_next_key(fd, nullptr, &next) == 0);
	REQUIRE(next == 0);

	// An in-range key advances by one.
	const uint32_t first = 0;
	REQUIRE(bpftime_map_get_next_key(fd, &first, &next) == 0);
	REQUIRE(next == 1);

	// An out-of-range key restarts at 0 rather than reporting -EINVAL.
	const uint32_t past_end = 99;
	next = 0xffffffffu;
	errno = 0;
	REQUIRE(bpftime_map_get_next_key(fd, &past_end, &next) == 0);
	REQUIRE(next == 0);
	REQUIRE(errno == 0);

	// The last index is the only one that ends the walk.
	const uint32_t last = 3;
	next = 0xffffffffu;
	errno = 0;
	REQUIRE(bpftime_map_get_next_key(fd, &last, &next) == -1);
	REQUIRE(errno == ENOENT);
}
} // namespace

TEST_CASE("prog array get_next_key follows kernel array semantics",
	  "[prog_array][map_iteration]")
{
	shm_remove remover("BPFTIME_PROG_ARRAY_ITER_TEST_SHM");
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);

	create_map_or_skip(prog_array_fd, "prog_array_iter",
			   BPF_MAP_TYPE_PROG_ARRAY, 4);
	create_map_or_skip(perf_array_fd, "perf_array_iter",
			   BPF_MAP_TYPE_PERF_EVENT_ARRAY, 4);

	SECTION("prog array")
	{
		// A sparse array: only slot 0 is populated. The walk still
		// visits 1..3, because array iteration does not skip holes.
		struct bpf_insn insn[] = {
			BPF_MOV64_IMM(0, 0),
			BPF_EXIT_INSN(),
		};
		REQUIRE(bpftime_progs_create(
				target_prog_fd, (const ebpf_inst *)insn,
				sizeof(insn) / sizeof(insn[0]), "iter_target",
				(int)BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
		const uint32_t key0 = 0;
		REQUIRE(bpftime_map_update_elem(prog_array_fd, &key0,
						 &target_prog_fd, 0) == 0);
		check_array_iteration(prog_array_fd);
	}

	SECTION("perf event array")
	{
		check_array_iteration(perf_array_fd);
	}
}

#endif