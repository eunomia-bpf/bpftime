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
#include "user/bpf_prog_insns.hpp"
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

TEST_CASE("Daemon rejects instruction counts beyond the captured buffer")
{
	bpf_insn_data captured = {};
	captured.code_len = BPF_COMPLEXITY_LIMIT_INSNS + 1;
	std::vector<ebpf_inst> insns(1);

	const int result = bpftime::detail::copy_captured_bpf_prog_insns(
		insns, captured);
	REQUIRE(result == -E2BIG);
	REQUIRE(insns.empty());
}

TEST_CASE("Daemon accepts the exact captured instruction capacity")
{
	bpf_insn_data captured = {};
	captured.code_len = BPF_COMPLEXITY_LIMIT_INSNS;
	auto *captured_insns = reinterpret_cast<ebpf_inst *>(captured.code);
	captured_insns[0].code = EBPF_OP_MOV64_IMM;
	captured_insns[0].imm = 7;
	captured_insns[BPF_COMPLEXITY_LIMIT_INSNS - 1].code = EBPF_OP_EXIT;
	std::vector<ebpf_inst> insns;

	REQUIRE(bpftime::detail::copy_captured_bpf_prog_insns(insns,
							     captured) == 0);
	REQUIRE(insns.size() == BPF_COMPLEXITY_LIMIT_INSNS);
	REQUIRE(insns.front().code == EBPF_OP_MOV64_IMM);
	REQUIRE(insns.front().imm == 7);
	REQUIRE(insns.back().code == EBPF_OP_EXIT);
}
