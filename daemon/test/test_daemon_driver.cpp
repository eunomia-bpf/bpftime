/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "handler/link_handler.hpp"
#include "user/bpftime_driver.hpp"
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif

TEST_CASE("Daemon driver preserves perf-link cookies")
{
	constexpr int server_pid = 1234;
	constexpr int perf_fd = 7;
	constexpr int prog_fd = 8;
	constexpr uint64_t cookie = 0x12345678;
	bpftime::bpftime_driver driver({}, nullptr);
	ebpf_inst insn = {};

	REQUIRE(bpftime_progs_create(
			prog_fd, &insn, 1, "cookie_test",
			static_cast<int>(
				bpftime::bpf_prog_type::BPF_PROG_TYPE_KPROBE)) >=
		0);
	REQUIRE(driver.bpftime_uprobe_create_server(server_pid, perf_fd, 0,
						    "/proc/self/exe", 0, false,
						    0) >= 0);

	REQUIRE(driver.bpftime_attach_perf_to_bpf_server(server_pid, perf_fd,
							 prog_fd, cookie) >= 0);
	const auto *manager =
		bpftime::shm_holder.global_shared_memory.get_manager();
	bool found = false;
	for (std::size_t i = 0; i < manager->size(); i++) {
		if (auto *link = std::get_if<bpftime::bpf_link_handler>(
			    &manager->get_handler(i));
		    link != nullptr && link->attach_cookie == cookie) {
			found = true;
		}
	}
	REQUIRE(found);
}
