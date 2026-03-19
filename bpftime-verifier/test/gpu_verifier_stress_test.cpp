#include <bpftime-verifier.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <config.hpp>
#include <ebpf_vm_isa.hpp>
#include <gpu_platform.hpp>
#include <gpu_verifier.hpp>
#include <resource_budget.hpp>
#include <simt_safety_check.hpp>
#include <uniformity_analysis.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <vector>

#include <linux/bpf.h>

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;
using Catch::Matchers::ContainsSubstring;

namespace
{

constexpr int32_t TEST_MAP_FD = 2333;
constexpr uint8_t OP_MOV64_IMM = INST_CLS_ALU64 | INST_ALU_OP_MOV;
constexpr uint8_t OP_MOV64_REG =
	INST_CLS_ALU64 | INST_ALU_OP_MOV | INST_SRC_REG;
constexpr uint8_t OP_ADD64_IMM = INST_CLS_ALU64 | INST_ALU_OP_ADD;
constexpr uint8_t OP_LDXDW = INST_CLS_LDX | 0x60 | INST_SIZE_DW;
constexpr uint8_t OP_STXH = INST_CLS_STX | 0x60 | INST_SIZE_H;
constexpr uint8_t OP_STXDW = INST_CLS_STX | 0x60 | INST_SIZE_DW;
constexpr uint8_t OP_JEQ_IMM = INST_CLS_JMP | 0x10;

ebpf_inst make_mov64_imm(uint8_t dst_reg, int32_t imm)
{
	ebpf_inst insn{};
	insn.opcode = OP_MOV64_IMM;
	insn.dst = dst_reg;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_mov64_reg(uint8_t dst_reg, uint8_t src_reg)
{
	ebpf_inst insn{};
	insn.opcode = OP_MOV64_REG;
	insn.dst = dst_reg;
	insn.src = src_reg;
	return insn;
}

ebpf_inst make_add64_imm(uint8_t dst_reg, int32_t imm)
{
	ebpf_inst insn{};
	insn.opcode = OP_ADD64_IMM;
	insn.dst = dst_reg;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_call(int32_t helper_id)
{
	ebpf_inst insn{};
	insn.opcode = INST_OP_CALL;
	insn.imm = helper_id;
	return insn;
}

ebpf_inst make_ja(int16_t off)
{
	ebpf_inst insn{};
	insn.opcode = INST_OP_JA;
	insn.offset = off;
	return insn;
}

ebpf_inst make_jeq_imm(uint8_t dst_reg, int32_t imm, int16_t off)
{
	ebpf_inst insn{};
	insn.opcode = OP_JEQ_IMM;
	insn.dst = dst_reg;
	insn.offset = off;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_stxdw(uint8_t dst_reg, uint8_t src_reg, int16_t off)
{
	ebpf_inst insn{};
	insn.opcode = OP_STXDW;
	insn.dst = dst_reg;
	insn.src = src_reg;
	insn.offset = off;
	return insn;
}

ebpf_inst make_stxh(uint8_t dst_reg, uint8_t src_reg, int16_t off)
{
	ebpf_inst insn{};
	insn.opcode = OP_STXH;
	insn.dst = dst_reg;
	insn.src = src_reg;
	insn.offset = off;
	return insn;
}

ebpf_inst make_ldxdw(uint8_t dst_reg, uint8_t src_reg, int16_t off)
{
	ebpf_inst insn{};
	insn.opcode = OP_LDXDW;
	insn.dst = dst_reg;
	insn.src = src_reg;
	insn.offset = off;
	return insn;
}

ebpf_inst make_exit()
{
	ebpf_inst insn{};
	insn.opcode = INST_OP_EXIT;
	return insn;
}

void append_lddw_map_fd(std::vector<ebpf_inst> &program, uint8_t dst_reg,
			int32_t map_fd)
{
	ebpf_inst insn{};
	insn.opcode = INST_OP_LDDW_IMM;
	insn.dst = dst_reg;
	insn.src = 1;
	insn.imm = map_fd;
	program.push_back(insn);
	program.push_back(ebpf_inst{});
}

void reset_verifier_state(std::initializer_list<int32_t> helpers = {},
			  const std::map<int, BpftimeMapDescriptor> &maps = {})
{
	set_available_helpers(std::vector<int32_t>(helpers));
	set_non_kernel_helpers(std::map<int32_t, BpftimeHelperProrotype>{});
	set_map_descriptors(maps);
}

BpftimeHelperProrotype make_helper_prototype(
	const char *name, bpftime_return_type_t return_type,
	std::initializer_list<bpftime_argument_type_t> argument_types)
{
	BpftimeHelperProrotype prototype{};
	prototype.name = name;
	prototype.return_type = return_type;
	for (size_t i = 0; i < 5; ++i) {
		prototype.argument_type[i] =
			bpftime::verifier::EBPF_ARGUMENT_TYPE_DONTCARE;
	}

	size_t index = 0;
	for (const auto argument_type : argument_types) {
		prototype.argument_type[index++] = argument_type;
	}
	return prototype;
}

std::map<int, BpftimeMapDescriptor>
make_test_map_descriptors(unsigned int key_size = 8)
{
	return {
		{ TEST_MAP_FD, BpftimeMapDescriptor{ .original_fd = TEST_MAP_FD,
						     .type = BPF_MAP_TYPE_HASH,
						     .key_size = key_size,
						     .value_size = 8,
						     .max_entries = 64,
						     .inner_map_fd = 0 } },
	};
}

template <typename Predicate>
const GpuHelperPrototype *find_gpu_helper(Predicate predicate)
{
	for (int32_t helper_id = 1; helper_id <= 4096; ++helper_id) {
		if (const auto *helper = find_gpu_helper_prototype(helper_id);
		    helper != nullptr && predicate(*helper)) {
			return helper;
		}
	}
	return nullptr;
}

bool has_semantic_args(const GpuHelperPrototype &helper)
{
	return std::any_of(helper.semantic_argument_types.begin(),
			   helper.semantic_argument_types.end(),
			   [](GpuHelperArgumentSemantics semantics) {
				   return semantics !=
					  GpuHelperArgumentSemantics::NONE;
			   });
}

std::vector<ebpf_inst>
make_map_update_program_from_scalar_helper(int32_t scalar_helper_id)
{
	std::vector<ebpf_inst> program;
	program.reserve(24);

	if (const auto *helper = find_gpu_helper_prototype(scalar_helper_id);
	    helper != nullptr && has_semantic_args(*helper)) {
		program.push_back(make_mov64_reg(1, 10));
		program.push_back(make_add64_imm(1, -8));
		program.push_back(make_mov64_reg(2, 10));
		program.push_back(make_add64_imm(2, -16));
		program.push_back(make_mov64_reg(3, 10));
		program.push_back(make_add64_imm(3, -24));
		program.push_back(make_call(scalar_helper_id));
		program.push_back(make_mov64_reg(6, 10));
		program.push_back(make_add64_imm(6, -8));
		program.push_back(make_ldxdw(0, 6, 0));
	} else {
		program.push_back(make_mov64_imm(1, 0));
		program.push_back(make_mov64_imm(2, 0));
		program.push_back(make_mov64_imm(3, 0));
		program.push_back(make_call(scalar_helper_id));
	}
	program.push_back(make_stxdw(10, 0, -8));
	program.push_back(make_mov64_imm(0, 1));
	program.push_back(make_stxdw(10, 0, -16));
	program.push_back(make_mov64_reg(2, 10));
	program.push_back(make_add64_imm(2, -8));
	program.push_back(make_mov64_reg(3, 10));
	program.push_back(make_add64_imm(3, -16));
	program.push_back(make_mov64_imm(4, 0));
	append_lddw_map_fd(program, 1, TEST_MAP_FD);
	program.push_back(make_call(2));
	program.push_back(make_exit());
	return program;
}

GpuVerifyResult
verify_map_update_program_from_scalar_helper(int32_t scalar_helper_id)
{
	reset_verifier_state({ scalar_helper_id, 2 },
			     make_test_map_descriptors());
	const auto program =
		make_map_update_program_from_scalar_helper(scalar_helper_id);
	return verify_gpu_program(program.data(), program.size(),
				  "cuda__stress_map_update");
}

GpuVerifyResult verify_program(
	const std::vector<ebpf_inst> &program,
	std::initializer_list<int32_t> helpers = {},
	const std::map<int, BpftimeMapDescriptor> &maps = {},
	const std::map<int32_t, BpftimeHelperProrotype> &non_kernel_helpers = {},
	const char *section_name = "cuda__stress")
{
	reset_verifier_state(helpers, maps);
	if (!non_kernel_helpers.empty()) {
		set_non_kernel_helpers(non_kernel_helpers);
	}
	return verify_gpu_program(program.data(), program.size(), section_name);
}

std::vector<ebpf_inst>
make_map_lookup_program_with_split_key(bool vary_last_half)
{
	std::vector<ebpf_inst> program;
	program.reserve(12);
	program.push_back(make_mov64_imm(0, 1));
	program.push_back(make_stxdw(10, 0, -8));
	if (vary_last_half) {
		program.push_back(make_call(511));
		program.push_back(make_stxh(10, 0, -6));
	}
	program.push_back(make_mov64_reg(2, 10));
	program.push_back(make_add64_imm(2, -8));
	append_lddw_map_fd(program, 1, TEST_MAP_FD);
	program.push_back(make_call(1));
	program.push_back(make_exit());
	return program;
}

std::vector<ebpf_inst> make_map_lookup_value_load_program(bool vary_last_half)
{
	auto program = make_map_lookup_program_with_split_key(vary_last_half);
	program.pop_back();
	program.push_back(make_ldxdw(6, 0, 0));
	program.push_back(make_exit());
	return program;
}

std::vector<ebpf_inst>
make_map_update_program_with_split_key(bool vary_last_half)
{
	std::vector<ebpf_inst> program;
	program.reserve(16);
	program.push_back(make_mov64_imm(0, 1));
	program.push_back(make_stxdw(10, 0, -8));
	if (vary_last_half) {
		program.push_back(make_call(511));
		program.push_back(make_stxh(10, 0, -6));
	}
	program.push_back(make_mov64_imm(0, 7));
	program.push_back(make_stxdw(10, 0, -16));
	program.push_back(make_mov64_reg(2, 10));
	program.push_back(make_add64_imm(2, -8));
	program.push_back(make_mov64_reg(3, 10));
	program.push_back(make_add64_imm(3, -16));
	program.push_back(make_mov64_imm(4, 0));
	append_lddw_map_fd(program, 1, TEST_MAP_FD);
	program.push_back(make_call(2));
	program.push_back(make_exit());
	return program;
}

std::vector<ebpf_inst> make_path_merged_map_update_value_program()
{
	std::vector<ebpf_inst> program;
	program.reserve(32);

	program.push_back(make_mov64_imm(0, 7));
	program.push_back(make_stxdw(10, 0, -8));
	program.push_back(make_mov64_reg(2, 10));
	program.push_back(make_add64_imm(2, -8));
	append_lddw_map_fd(program, 1, TEST_MAP_FD);
	program.push_back(make_call(1));
	program.push_back(make_jeq_imm(0, 0, 5));

	program.push_back(make_mov64_imm(4, 2));
	program.push_back(make_stxdw(10, 4, -16));
	program.push_back(make_mov64_reg(3, 10));
	program.push_back(make_add64_imm(3, -16));
	program.push_back(make_ja(4));

	program.push_back(make_mov64_imm(4, 1));
	program.push_back(make_stxdw(10, 4, -24));
	program.push_back(make_mov64_reg(3, 10));
	program.push_back(make_add64_imm(3, -24));

	program.push_back(make_mov64_imm(4, 0));
	program.push_back(make_mov64_reg(2, 10));
	program.push_back(make_add64_imm(2, -8));
	append_lddw_map_fd(program, 1, TEST_MAP_FD);
	program.push_back(make_call(2));
	program.push_back(make_exit());
	return program;
}

} // namespace

TEST_CASE("G5 helper metadata drives GPU SIMT behavior", "[gpu][stress][g5]")
{
	SECTION("A metadata-selected varying helper produces VARYING R0")
	{
		const auto *helper =
			find_gpu_helper([](const GpuHelperPrototype &it) {
				return it.return_uniformity ==
					       GpuHelperUniformity::VARYING &&
				       it.behavior ==
					       GpuHelperBehavior::GENERIC &&
				       it.effect_class ==
					       GpuHelperEffectClass::NONE &&
				       !has_semantic_args(it);
			});
		REQUIRE(helper != nullptr);

		reset_verifier_state({ helper->id });
		const std::vector<ebpf_inst> program = {
			make_call(helper->id),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states.size() == program.size());
		REQUIRE(result.states[1].regs[0] == Uniformity::VARYING);
	}

	SECTION("A metadata-selected prohibited helper is rejected by SIMT safety")
	{
		const auto *helper =
			find_gpu_helper([](const GpuHelperPrototype &it) {
				return it.effect_class ==
				       GpuHelperEffectClass::PROHIBITED_SYNC;
			});
		REQUIRE(helper != nullptr);

		reset_verifier_state({ helper->id });
		const std::vector<ebpf_inst> program = {
			make_call(helper->id),
			make_exit(),
		};

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.prohibited_helper_count == 1);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name == "Prohibited Helpers");
	}

	SECTION("A warp-aggregation helper returns UNIFORM despite varying inputs")
	{
		const auto *helper =
			find_gpu_helper([](const GpuHelperPrototype &it) {
				return it.effect_class ==
				       GpuHelperEffectClass::WARP_AGGREGATION;
			});
		REQUIRE(helper != nullptr);

		reset_verifier_state({ 505, helper->id });
		const std::vector<ebpf_inst> program = {
			make_call(505),
			make_mov64_reg(1, 0),
			make_call(helper->id),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[3].regs[0] == Uniformity::UNIFORM);
	}
}

TEST_CASE("Dead code is excluded from SIMT safety checks",
	  "[gpu][stress][dead-code]")
{
	SECTION("unreachable CALL 506 is ignored")
	{
		const std::vector<ebpf_inst> program = {
			make_mov64_imm(0, 0),
			make_ja(1),
			make_call(506),
			make_exit(),
		};

		reset_verifier_state({ 506 });
		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);
		REQUIRE(uniformity.reachable.size() == program.size());
		REQUIRE(uniformity.reachable[0]);
		REQUIRE(uniformity.reachable[1]);
		REQUIRE_FALSE(uniformity.reachable[2]);
		REQUIRE(uniformity.reachable[3]);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE(safety.passed);

		const auto result = verify_program(program, { 506 }, {}, {},
						   "cuda__stress_dead_call");
		INFO(result.error_message);
		REQUIRE(result.passed);
	}

	SECTION("unreachable varying branch is ignored")
	{
		const std::vector<ebpf_inst> program = {
			make_mov64_imm(0, 0), make_ja(3),
			make_call(505),	      make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1), make_exit(),
		};

		reset_verifier_state({ 505 });
		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);
		REQUIRE(uniformity.reachable.size() == program.size());
		REQUIRE(uniformity.reachable[0]);
		REQUIRE(uniformity.reachable[1]);
		REQUIRE_FALSE(uniformity.reachable[2]);
		REQUIRE_FALSE(uniformity.reachable[3]);
		REQUIRE_FALSE(uniformity.reachable[4]);
		REQUIRE(uniformity.reachable[5]);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE(safety.passed);

		const auto result = verify_program(program, { 505 }, {}, {},
						   "cuda__stress_dead_branch");
		INFO(result.error_message);
		REQUIRE(result.passed);
	}
}

TEST_CASE(
	"G1 stack uniformity tracks stack-resident values and helper out-params",
	"[gpu][stress][g1]")
{
	SECTION("thread_idx stored to stack and loaded back stays VARYING")
	{
		reset_verifier_state({ 505 });
		const std::vector<ebpf_inst> program = {
			make_call(505),	       make_stxdw(10, 0, -8),
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_ldxdw(2, 1, 0),   make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[2], -8, 8) ==
			Uniformity::VARYING);
		REQUIRE(result.states[5].regs[2] == Uniformity::VARYING);
	}

	SECTION("uniform scalar stored to stack and loaded back stays UNIFORM")
	{
		reset_verifier_state();
		const std::vector<ebpf_inst> program = {
			make_mov64_imm(0, 7),  make_stxdw(10, 0, -8),
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_ldxdw(2, 1, 0),   make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[2], -8, 8) ==
			Uniformity::UNIFORM);
		REQUIRE(result.states[5].regs[2] == Uniformity::UNIFORM);
	}

	SECTION("thread_idx helper out-param marks stack bytes VARYING")
	{
		reset_verifier_state({ 505 });
		const std::vector<ebpf_inst> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_imm(2, 0),  make_mov64_imm(3, 0),
			make_call(505),	       make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[5], -8, 8) ==
			Uniformity::VARYING);
	}

	SECTION("thread_idx helper writes all three stack out-params as VARYING")
	{
		reset_verifier_state({ 505 });
		const std::vector<ebpf_inst> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_reg(2, 10), make_add64_imm(2, -16),
			make_mov64_reg(3, 10), make_add64_imm(3, -24),
			make_call(505),	       make_mov64_reg(4, 10),
			make_add64_imm(4, -8), make_ldxdw(5, 4, 0),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[7], -8, 8) ==
			Uniformity::VARYING);
		REQUIRE(query_stack_uniformity(result.states[7], -16, 8) ==
			Uniformity::VARYING);
		REQUIRE(query_stack_uniformity(result.states[7], -24, 8) ==
			Uniformity::VARYING);
		REQUIRE(result.states[10].regs[5] == Uniformity::VARYING);
	}

	SECTION("block_idx helper out-param marks stack bytes UNIFORM")
	{
		reset_verifier_state({ 503 });
		const std::vector<ebpf_inst> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_imm(2, 0),  make_mov64_imm(3, 0),
			make_call(503),	       make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[5], -8, 8) ==
			Uniformity::UNIFORM);
	}

	SECTION("block_idx helper out-param reload stays UNIFORM")
	{
		reset_verifier_state({ 503 });
		const std::vector<ebpf_inst> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_imm(2, 0),  make_mov64_imm(3, 0),
			make_call(503),	       make_mov64_reg(4, 10),
			make_add64_imm(4, -8), make_ldxdw(5, 4, 0),
			make_exit(),
		};

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(query_stack_uniformity(result.states[5], -8, 8) ==
			Uniformity::UNIFORM);
		REQUIRE(result.states[8].regs[5] == Uniformity::UNIFORM);
	}

	SECTION("branching on a value reloaded from stack after thread_idx is rejected")
	{
		reset_verifier_state({ 505 });
		const std::vector<ebpf_inst> program = {
			make_call(505),	       make_stxdw(10, 0, -8),
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_ldxdw(0, 1, 0),   make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1),  make_exit(),
		};

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.varying_branch_count == 1);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name ==
			"Warp-Uniform Branch Conditions");
	}
}

TEST_CASE("Unknown GPU helpers are modeled conservatively",
	  "[gpu][stress][unknown-helper]")
{
	const std::vector<ebpf_inst> program = {
		make_call(600),
		make_jeq_imm(0, 0, 1),
		make_mov64_imm(0, 1),
		make_exit(),
	};
	const std::map<int32_t, BpftimeHelperProrotype> helper_prototypes = {
		{ 600,
		  make_helper_prototype(
			  "bpf_unknown_gpu_helper",
			  bpftime::verifier::EBPF_RETURN_TYPE_INTEGER, {}) },
	};

	const auto result =
		verify_program(program, { 600 }, {}, helper_prototypes,
			       "cuda__stress_unknown_helper");
	INFO(result.error_message);
	REQUIRE_FALSE(result.passed);
	REQUIRE_THAT(result.error_message,
		     ContainsSubstring("Warp-Uniform Branch Conditions"));
}

TEST_CASE("G3 full-width key tracking covers map_lookup and map_update",
	  "[gpu][stress][g3][map-key-width]")
{
	SECTION("map_lookup return becomes VARYING when later key bytes vary")
	{
		reset_verifier_state({ 1, 511 }, make_test_map_descriptors(4));
		const auto program =
			make_map_lookup_program_with_split_key(true);

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states.back().regs[0] == Uniformity::VARYING);
	}

	SECTION("map_lookup return stays UNIFORM when all key bytes are uniform")
	{
		reset_verifier_state({ 1 }, make_test_map_descriptors(4));
		const auto program =
			make_map_lookup_program_with_split_key(false);

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states.back().regs[0] == Uniformity::UNIFORM);
	}

	SECTION("map value loads inherit varying key uniformity")
	{
		reset_verifier_state({ 1, 511 }, make_test_map_descriptors(4));
		const auto program = make_map_lookup_value_load_program(true);

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[program.size() - 1].regs[6] ==
			Uniformity::VARYING);
	}

	SECTION("map value loads stay UNIFORM for uniform keys")
	{
		reset_verifier_state({ 1 }, make_test_map_descriptors(4));
		const auto program = make_map_lookup_value_load_program(false);

		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[program.size() - 1].regs[6] ==
			Uniformity::UNIFORM);
	}

	SECTION("map_update rejects a four-byte key when only the last half varies")
	{
		const auto program =
			make_map_update_program_with_split_key(true);
		const auto result =
			verify_program(program, { 2, 511 },
				       make_test_map_descriptors(4), {},
				       "cuda__stress_split_key_reject");
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("Map Update Key Uniformity"));
	}

	SECTION("map_update accepts a four-byte key when all bytes are uniform")
	{
		const auto program =
			make_map_update_program_with_split_key(false);
		const auto result =
			verify_program(program, { 2 },
				       make_test_map_descriptors(4), {},
				       "cuda__stress_split_key_pass");
		INFO(result.error_message);
		REQUIRE(result.passed);
	}

	SECTION("map_update without a descriptor still rejects varying stored key bytes")
	{
		reset_verifier_state({ 2, 511 });
		const auto program =
			make_map_update_program_with_split_key(true);

		const auto uniformity =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.varying_map_key_count == 1);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name ==
			"Map Update Key Uniformity");
	}
}

TEST_CASE("G3 map_update checks key bytes, not just pointer uniformity",
	  "[gpu][stress][g3]")
{
	SECTION("map_update rejects a key loaded from stack bytes written by thread_idx")
	{
		const auto result =
			verify_map_update_program_from_scalar_helper(505);
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("Map Update Key Uniformity"));
	}

	SECTION("map_update accepts a key loaded from stack bytes written by block_idx")
	{
		const auto result =
			verify_map_update_program_from_scalar_helper(503);
		INFO(result.error_message);
		REQUIRE(result.passed);
	}

	SECTION("map_update rejects a key derived from lane_id stored on stack")
	{
		const auto result =
			verify_map_update_program_from_scalar_helper(511);
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("Map Update Key Uniformity"));
	}
}

TEST_CASE("G4 resource budget uses worst-case counts", "[gpu][stress][g4]")
{
	SECTION("simple program within budget passes")
	{
		const std::vector<ebpf_inst> program = {
			make_mov64_imm(0, 1),
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
		REQUIRE(result.passed);
		REQUIRE(result.instruction_count == program.size());
	}

	SECTION("program exceeding instruction budget is rejected")
	{
		const std::vector<ebpf_inst> program = {
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
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("instruction count"));
	}

	SECTION("loop-aware helper budget rejects when verifier stats enlarge worst-case")
	{
		const std::vector<ebpf_inst> program = {
			make_call(14),
			make_exit(),
		};
		const GpuResourceBudget budget{
			.max_instructions = 64,
			.max_helper_calls = 14,
			.max_memory_ops = 0,
			.max_map_lookups = 0,
			.max_map_updates = 0,
		};
		const ebpf_verifier_stats_t stats{
			.total_unreachable = 0,
			.total_warnings = 0,
			.max_instruction_count = 30,
		};

		const auto result = check_resource_budget(
			program.data(), program.size(), budget, &stats);
		REQUIRE_FALSE(result.passed);
		REQUIRE(result.instruction_count == 30);
		REQUIRE(result.helper_call_count == 15);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("helper call count"));
	}

	SECTION("verifier stats never shrink static helper accounting")
	{
		const std::vector<ebpf_inst> program = {
			make_call(14),
			make_call(14),
			make_exit(),
		};
		const GpuResourceBudget budget{
			.max_instructions = 8,
			.max_helper_calls = 2,
			.max_memory_ops = 0,
			.max_map_lookups = 0,
			.max_map_updates = 0,
		};
		const ebpf_verifier_stats_t stats{
			.total_unreachable = 0,
			.total_warnings = 0,
			.max_instruction_count = 1,
		};

		const auto result = check_resource_budget(
			program.data(), program.size(), budget, &stats);
		REQUIRE(result.passed);
		REQUIRE(result.instruction_count == 3);
		REQUIRE(result.helper_call_count == 2);
	}
}

TEST_CASE("G2 context loads are uniform only for the unmodified context base",
	  "[gpu][stress][g2]")
{
	SECTION("direct context load from R1 is UNIFORM")
	{
		const std::vector<ebpf_inst> program = {
			make_ldxdw(0, 1, 0),
			make_exit(),
		};

		reset_verifier_state();
		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::UNIFORM);
	}

	SECTION("modified context pointer loads are conservative")
	{
		const std::vector<ebpf_inst> program = {
			make_mov64_reg(2, 1),
			make_add64_imm(2, 8),
			make_ldxdw(0, 2, 0),
			make_exit(),
		};

		reset_verifier_state();
		const auto result =
			analyze_uniformity(program.data(), program.size());
		REQUIRE(result.success);
		REQUIRE(result.states[3].regs[0] == Uniformity::VARYING);
	}
}

TEST_CASE("G1 and G3 compose across stack-to-map-key flows",
	  "[gpu][stress][g1][g3]")
{
	SECTION("thread_idx to stack to map_update is rejected")
	{
		const auto result =
			verify_map_update_program_from_scalar_helper(505);
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("Map Update Key Uniformity"));
	}

	SECTION("block_idx to stack to map_update is accepted")
	{
		const auto result =
			verify_map_update_program_from_scalar_helper(503);
		INFO(result.error_message);
		REQUIRE(result.passed);
	}
}

TEST_CASE("Path-merged map_update value pointers work without PREVAIL",
	  "[gpu][stress][prevail]")
{
	const auto maps = make_test_map_descriptors();
	reset_verifier_state({ 1, 2 }, maps);
	const auto program = make_path_merged_map_update_value_program();

	SECTION("SIMT-only mode still accepts the program")
	{
		GpuVerifierConfig config;
		config.skip_prevail = true;
		config.map_descriptors = maps;

		const auto result =
			verify_gpu_program(program.data(), program.size(),
					   "cuda__stress_prevail_merge",
					   config);
		INFO(result.error_message);
		REQUIRE(result.passed);
		REQUIRE(result.prevail_time_us == 0.0);
	}

	SECTION("PREVAIL_ONLY reports the current path-merge limitation")
	{
		GpuVerifierConfig config;
		config.mode = GpuVerificationMode::PREVAIL_ONLY;
		config.skip_prevail = false;
		config.map_descriptors = maps;

		const auto result =
			verify_gpu_program(program.data(), program.size(),
					   "cuda__stress_prevail_merge",
					   config);
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring("Illegal map update"));
	}
}
