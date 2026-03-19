#include <bpftime-verifier.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ebpf_inst.h>
#include <gpu_verifier.hpp>
#include <resource_budget.hpp>
#include <simt_safety_check.hpp>
#include <uniformity_analysis.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;

namespace
{

ebpf_inst make_mov64_imm(uint8_t dst_reg, int32_t imm)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_MOV64_IMM;
	insn.dst_reg = dst_reg;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_mov64_reg(uint8_t dst_reg, uint8_t src_reg)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_MOV64_REG;
	insn.dst_reg = dst_reg;
	insn.src_reg = src_reg;
	return insn;
}

ebpf_inst make_add64_imm(uint8_t dst_reg, int32_t imm)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_ADD64_IMM;
	insn.dst_reg = dst_reg;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_call(int32_t helper_id)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_CALL;
	insn.imm = helper_id;
	return insn;
}

ebpf_inst make_jeq_imm(uint8_t dst_reg, int32_t imm, int16_t off)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_JEQ_IMM;
	insn.dst_reg = dst_reg;
	insn.off = off;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_ldxdw(uint8_t dst_reg, uint8_t src_reg, int16_t off)
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_LDXDW;
	insn.dst_reg = dst_reg;
	insn.src_reg = src_reg;
	insn.off = off;
	return insn;
}

ebpf_inst make_exit()
{
	ebpf_inst insn{};
	insn.code = EBPF_OP_EXIT;
	return insn;
}

void reset_verifier_state(std::initializer_list<int32_t> helpers = {})
{
	set_available_helpers(std::vector<int32_t>(helpers));
	set_non_kernel_helpers(std::map<int32_t, BpftimeHelperProrotype>{});
	set_map_descriptors(std::map<int, BpftimeMapDescriptor>{});
}

} // namespace

TEST_CASE("Uniformity analysis classifies constants and GPU helpers",
	  "[gpu][uniformity]")
{
	SECTION("constant is UNIFORM")
	{
		reset_verifier_state();
		const std::array<ebpf_inst, 2> program = {
			make_mov64_imm(0, 7),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states.size() == program.size());
		REQUIRE(result.states[1].regs[0] == Uniformity::UNIFORM);
	}

	SECTION("thread_idx helper 505 makes R0 VARYING")
	{
		reset_verifier_state({ 505 });
		const std::array<ebpf_inst, 2> program = {
			make_call(505),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::VARYING);
	}

	SECTION("block_idx helper 503 is UNIFORM")
	{
		reset_verifier_state({ 503 });
		const std::array<ebpf_inst, 2> program = {
			make_call(503),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::UNIFORM);
	}
}

TEST_CASE("SIMT safety enforces uniform branches and helper restrictions",
	  "[gpu][simt]")
{
	SECTION("uniform branch passes")
	{
		reset_verifier_state();
		const std::array<ebpf_inst, 4> program = {
			make_mov64_imm(0, 1),
			make_jeq_imm(0, 1, 1),
			make_mov64_imm(0, 0),
			make_exit(),
		};

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE(safety.passed);
	}

	SECTION("varying branch from thread_idx is rejected")
	{
		reset_verifier_state({ 505 });
		const std::array<ebpf_inst, 4> program = {
			make_call(505),
			make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1),
			make_exit(),
		};

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name ==
			"Warp-Uniform Branch Conditions");
	}

	SECTION("prohibited helper 506 is rejected")
	{
		reset_verifier_state({ 506 });
		const std::array<ebpf_inst, 2> program = {
			make_call(506),
			make_exit(),
		};

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name == "Prohibited Helpers");
	}
}

TEST_CASE("Resource budget tracks instruction limits", "[gpu][budget]")
{
	reset_verifier_state();

	SECTION("within budget passes")
	{
		const std::array<ebpf_inst, 2> program = {
			make_mov64_imm(0, 1),
			make_exit(),
		};
		const GpuResourceBudget budget{
			.max_instructions = 2,
			.max_helper_calls = 1,
			.max_memory_ops = 0,
			.max_map_lookups = 0,
			.max_map_updates = 0,
		};

		const auto result = check_resource_budget(
			program.data(), program.size(), budget);
		REQUIRE(result.passed);
		REQUIRE(result.instruction_count == program.size());
	}

	SECTION("exceeding instruction count fails")
	{
		const std::array<ebpf_inst, 3> program = {
			make_mov64_imm(0, 1),
			make_mov64_imm(0, 2),
			make_exit(),
		};
		const GpuResourceBudget budget{
			.max_instructions = 2,
			.max_helper_calls = 0,
			.max_memory_ops = 0,
			.max_map_lookups = 0,
			.max_map_updates = 0,
		};

		const auto result = check_resource_budget(
			program.data(), program.size(), budget);
		REQUIRE_FALSE(result.passed);
		REQUIRE(result.error_message.find("instruction count") !=
			std::string::npos);
	}
}

TEST_CASE("GPU verifier integrates SIMT phases with optional PREVAIL",
	  "[gpu][integration]")
{
	SECTION("simple safe program passes")
	{
		reset_verifier_state();
		const std::array<ebpf_inst, 2> program = {
			make_mov64_imm(0, 0),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.error_message);
		REQUIRE(result.passed);
	}

	SECTION("unsafe varying branch fails")
	{
		reset_verifier_state({ 505 });
		const std::array<ebpf_inst, 13> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_reg(2, 10), make_add64_imm(2, -16),
			make_mov64_reg(3, 10), make_add64_imm(3, -24),
			make_call(505),	       make_mov64_reg(1, 10),
			make_add64_imm(1, -8), make_ldxdw(0, 1, 0),
			make_jeq_imm(0, 0, 1), make_mov64_imm(0, 1),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE(result.varying_branch_count == 1);
		REQUIRE(result.error_message.find(
				"Warp-Uniform Branch Conditions") !=
			std::string::npos);
	}

	SECTION("default GPU verifier runs PREVAIL for helper out-params")
	{
		reset_verifier_state({ 503 });
		const std::array<ebpf_inst, 11> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_reg(2, 10), make_add64_imm(2, -16),
			make_mov64_reg(3, 10), make_add64_imm(3, -24),
			make_call(503),	       make_mov64_reg(1, 10),
			make_add64_imm(1, -8), make_ldxdw(0, 1, 0),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.error_message);
		REQUIRE(result.passed);
		REQUIRE(result.prevail_time_us > 0.0);
	}

	SECTION("PREVAIL failures reject GPU programs before SIMT")
	{
		reset_verifier_state();
		const std::array<ebpf_inst, 3> program = {
			make_mov64_imm(1, 0),
			make_ldxdw(0, 1, 0),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE(result.prevail_time_us > 0.0);
		REQUIRE(result.simt_time_us == 0.0);
		REQUIRE_FALSE(result.error_message.empty());
	}

	SECTION("PREVAIL_ONLY accepts registered GPU helpers")
	{
		reset_verifier_state({ 503 });
		const std::array<ebpf_inst, 11> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_reg(2, 10), make_add64_imm(2, -16),
			make_mov64_reg(3, 10), make_add64_imm(3, -24),
			make_call(503),	       make_mov64_reg(1, 10),
			make_add64_imm(1, -8), make_ldxdw(0, 1, 0),
			make_exit(),
		};

		GpuVerifierConfig config;
		config.mode = GpuVerificationMode::PREVAIL_ONLY;
		config.skip_prevail = false;
		const auto result =
			verify_gpu_program(program.data(), program.size(),
					   "cuda__integration", config);
		INFO(result.error_message);
		REQUIRE(result.passed);
		REQUIRE(result.prevail_time_us > 0.0);
		REQUIRE(result.simt_time_us == 0.0);
		REQUIRE(result.error_message.empty());
	}
}
