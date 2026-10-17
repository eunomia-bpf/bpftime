/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Regression test for #704: bpftime_prog_load must reject programs calling
// helpers that were never registered into the VM, instead of succeeding at
// load time and failing on every invocation at runtime.
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

using namespace bpftime;

namespace
{
// Walk a vector of instructions calling `emit` for each 0x85 (helper call)
// instruction. Kept local so the test does not depend on internal headers.
static std::vector<uint32_t>
collect_helper_calls(const std::vector<struct ebpf_inst> &insns)
{
	std::vector<uint32_t> helpers;
	for (size_t i = 0; i < insns.size(); ++i) {
		const auto &inst = insns[i];
		if (inst.code == 0x18) {
			// Second half of a LDDW is data, skip it.
			++i;
			continue;
		}
		if (inst.code == 0x85 && inst.src_reg == 0)
			helpers.push_back(static_cast<uint32_t>(inst.imm));
	}
	return helpers;
}
} // namespace

TEST_CASE("Reject programs calling unregistered helpers at load time",
	  "[helper][load]")
{
	// r0 = 0
	struct ebpf_inst mov_r0 = { .code = 0xb7, .dst_reg = 0,
				    .src_reg = 0, .off = 0, .imm = 0 };
	// BPF_FUNC_probe_read_kernel_str (115) is not part of any helper
	// group shipped with bpftime, so the VM never registers its
	// implementation for calls.
	struct ebpf_inst call_115 = { .code = 0x85, .dst_reg = 0,
				      .src_reg = 0, .off = 0,
				      .imm = 115 };
	struct ebpf_inst exit_insn = { .code = 0x95, .dst_reg = 0,
				       .src_reg = 0, .off = 0, .imm = 0 };

	SECTION("Unregistered helper must fail load") {
		bpftime::runtime_config config;
		config.set_vm_name("llvm");
		std::vector<struct ebpf_inst> insns = { mov_r0, call_115,
							exit_insn };
		bpftime_prog prog(insns.data(), insns.size(), "t_unreg_helper",
				  std::move(config));
		REQUIRE(collect_helper_calls(insns) ==
			std::vector<uint32_t>{ 115 });
		REQUIRE(prog.bpftime_prog_load(false) < 0);
	}

	SECTION("Registered helper must still load") {
		bpftime::runtime_config config;
		config.set_vm_name("llvm");
		// BPF_FUNC_get_prandom_u32 (7) is registered by
		// bpftime_helper_group::get_kernel_utils_helper_group().
		struct ebpf_inst call_7 = { .code = 0x85, .dst_reg = 0,
					    .src_reg = 0, .off = 0,
					    .imm = 7 };
		std::vector<struct ebpf_inst> insns = { mov_r0, call_7,
							exit_insn };
		bpftime_prog prog(insns.data(), insns.size(), "t_reg_helper",
				  std::move(config));
		REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
				.add_helper_group_to_prog(&prog) == 0);
		REQUIRE(collect_helper_calls(insns) ==
			std::vector<uint32_t>{ 7 });
		REQUIRE(prog.bpftime_prog_load(false) == 0);
	}
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
	SECTION("CUDA program calling device-side helper must still load") {
		// GPU programs (cuda__*) get their helpers (e.g. 503 =
		// bpf_get_block_idx) from the device trampoline, never from
		// this VM; the host-side audit must not reject them.
		bpftime::runtime_config config;
		config.set_vm_name("llvm");
		struct ebpf_inst call_503 = { .code = 0x85, .dst_reg = 0,
					      .src_reg = 0, .off = 0,
					      .imm = 503 };
		std::vector<struct ebpf_inst> insns = { mov_r0, call_503,
							exit_insn };
		bpftime_prog prog(insns.data(), insns.size(),
				  "t_cuda__unreg_helper", std::move(config));
		REQUIRE(collect_helper_calls(insns) ==
			std::vector<uint32_t>{ 503 });
		REQUIRE(prog.bpftime_prog_load(false) == 0);
	}
#endif
}
