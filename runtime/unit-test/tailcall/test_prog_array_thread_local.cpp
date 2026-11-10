/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <linux/filter.h>

#include <atomic>
#include <cerrno>
#include <thread>
#include <unit-test/common_def.hpp>

using namespace bpftime;

#if __linux__

namespace
{
constexpr int prog_array_fd = 1101;
constexpr int target_fd_a = 1102;
constexpr int target_fd_b = 1103;

void create_target_prog(int fd)
{
	struct bpf_insn insn[] = {
		BPF_MOV64_IMM(0, 0x1234),
		BPF_EXIT_INSN(),
	};
	REQUIRE(bpftime_progs_create(
			fd, (const ebpf_inst *)insn,
			sizeof(insn) / sizeof(insn[0]), "tls_target",
			(int)bpftime::bpf_prog_type::BPF_PROG_TYPE_SOCKET_FILTER) >=
		0);
}
} // namespace

TEST_CASE("prog array lookup keeps a per-thread result slot",
	  "[prog_array][tailcall][tls]")
{
	shm_remove remover("BPFTIME_PROG_ARRAY_TLS_TEST_SHM");
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	REQUIRE(bpftime_maps_create(prog_array_fd, "prog_array_tls",
				    bpf_map_attr{
					    .type = (int)bpf_map_type::
						    BPF_MAP_TYPE_PROG_ARRAY,
					    .key_size = 4,
					    .value_size = 4,
					    .max_ents = 4 }) >= 0);
	create_target_prog(target_fd_a);
	create_target_prog(target_fd_b);

	int key0 = 0;
	int key1 = 1;
	REQUIRE(bpftime_map_update_elem(prog_array_fd, &key0, &target_fd_a,
					0) >= 0);
	REQUIRE(bpftime_map_update_elem(prog_array_fd, &key1, &target_fd_b,
					0) >= 0);

	// Two threads look up different entries concurrently. Each must keep
	// observing its own fd through the thread-local result slot.
	std::atomic<bool> start{ false };
	std::atomic<bool> failed{ false };
	auto worker = [&](int key, int expected_fd) {
		while (!start.load(std::memory_order_acquire))
			std::this_thread::yield();
		void *stable_pointer = nullptr;
		for (int round = 0; round < 1000; ++round) {
			void *result =
				bpftime_map_lookup_elem(prog_array_fd, &key);
			if (result == nullptr ||
			    *static_cast<int32_t *>(result) != expected_fd) {
				failed.store(true, std::memory_order_release);
				return;
			}
			if (stable_pointer == nullptr)
				stable_pointer = result;
			else if (result != stable_pointer)
				failed.store(true,
					     std::memory_order_release);
		}
	};

	std::thread first(worker, key0, target_fd_a);
	std::thread second(worker, key1, target_fd_b);
	start.store(true, std::memory_order_release);
	first.join();
	second.join();
	REQUIRE_FALSE(failed.load(std::memory_order_acquire));

	// Out-of-range and empty entries keep their existing error behavior.
	int bad_key = 4;
	errno = 0;
	REQUIRE(bpftime_map_lookup_elem(prog_array_fd, &bad_key) == nullptr);
	REQUIRE(errno == EINVAL);
	int empty_key = 2;
	errno = 0;
	REQUIRE(bpftime_map_lookup_elem(prog_array_fd, &empty_key) == nullptr);
	REQUIRE(errno == ENOENT);

	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(shm_open_type::SHM_NO_CREATE);
}

#endif
