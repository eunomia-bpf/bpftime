#include <bpftime-verifier.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ebpf_vm_isa.hpp>
#include <gpu_platform.hpp>
#include <gpu_verifier.hpp>
#include <simt_safety_check.hpp>
#include <uniformity_analysis.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;

namespace
{

ebpf_inst make_instruction(uint8_t opcode, uint8_t dst_reg = 0,
			   uint8_t src_reg = 0, int16_t offset = 0,
			   int32_t imm = 0)
{
	ebpf_inst insn{};
	insn.opcode = opcode;
	insn.dst = dst_reg;
	insn.src = src_reg;
	insn.offset = offset;
	insn.imm = imm;
	return insn;
}

ebpf_inst make_mov64_imm(uint8_t dst_reg, int32_t imm)
{
	return make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_MOV,
				dst_reg, 0, 0, imm);
}

ebpf_inst make_mov64_reg(uint8_t dst_reg, uint8_t src_reg)
{
	return make_instruction(INST_CLS_ALU64 | INST_SRC_REG | INST_ALU_OP_MOV,
				dst_reg, src_reg);
}

ebpf_inst make_add64_imm(uint8_t dst_reg, int32_t imm)
{
	return make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_ADD,
				dst_reg, 0, 0, imm);
}

ebpf_inst make_sub64_imm(uint8_t dst_reg, int32_t imm)
{
	return make_instruction(INST_CLS_ALU64 | INST_SRC_IMM | INST_ALU_OP_SUB,
				dst_reg, 0, 0, imm);
}

ebpf_inst make_call(int32_t helper_id)
{
	return make_instruction(INST_OP_CALL, 0, 0, 0, helper_id);
}

ebpf_inst make_atomic_add(uint8_t dst_reg, uint8_t src_reg)
{
	return make_instruction(INST_CLS_STX | INST_SIZE_DW | (INST_XADD << 5),
				dst_reg, src_reg);
}

ebpf_inst make_atomic(uint8_t dst_reg, uint8_t src_reg, int32_t operation)
{
	return make_instruction(INST_CLS_STX | INST_SIZE_DW | (INST_XADD << 5),
				dst_reg, src_reg, 0, operation);
}

ebpf_inst make_stdw_imm(uint8_t dst_reg, int16_t off, int32_t imm)
{
	return make_instruction(INST_CLS_ST | INST_SIZE_DW | (INST_MEM << 5),
				dst_reg, 0, off, imm);
}

ebpf_inst make_stxdw(uint8_t dst_reg, uint8_t src_reg, int16_t off = 0)
{
	return make_instruction(INST_CLS_STX | INST_SIZE_DW | (INST_MEM << 5),
				dst_reg, src_reg, off);
}

ebpf_inst make_lddw_map(uint8_t dst_reg, int32_t fd)
{
	return make_instruction(INST_OP_LDDW_IMM, dst_reg, 1, 0, fd);
}

ebpf_inst make_jeq_imm(uint8_t dst_reg, int32_t imm, int16_t off)
{
	return make_instruction(INST_CLS_JMP | INST_SRC_IMM | 0x10, dst_reg, 0,
				off, imm);
}

ebpf_inst make_ldxdw(uint8_t dst_reg, uint8_t src_reg, int16_t off)
{
	return make_instruction(INST_CLS_LDX | INST_SIZE_DW | (INST_MEM << 5),
				dst_reg, src_reg, off);
}

ebpf_inst make_exit()
{
	return make_instruction(INST_OP_EXIT);
}

template <size_t N>
UniformityAnalysisResult
analyze_program_uniformity(const std::array<ebpf_inst, N> &program)
{
	static const std::map<int, BpftimeMapDescriptor> no_maps;
	return analyze_uniformity(program.data(), program.size(), no_maps);
}

BpftimeMapDescriptor make_map(int fd, uint32_t type)
{
	return BpftimeMapDescriptor{
		.original_fd = fd,
		.type = type,
		.key_size = 4,
		.value_size = 8,
		.max_entries = 16,
		.inner_map_fd = static_cast<unsigned int>(-1),
	};
}

} // namespace

TEST_CASE("GPU platform accepts only supported map types", "[gpu][platform]")
{
	for (const uint32_t type :
	     { 1501, 1502, 1503, 1504, 1512, 1513, 1527 }) {
		REQUIRE(try_get_gpu_map_type(type).has_value());
	}
	REQUIRE_FALSE(try_get_gpu_map_type(1505).has_value());

	const std::array<ebpf_inst, 2> program = {
		make_mov64_imm(0, 0),
		make_exit(),
	};
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, make_map(1, 1505) },
	};
	REQUIRE(verify_gpu_program(program.data(), program.size(),
				   "cuda__unknown_map", maps));
}

TEST_CASE("GPU verifier matches the no-context execution ABI", "[gpu][prevail]")
{
	SECTION("R1 context reads are rejected")
	{
		const std::array<ebpf_inst, 2> program = {
			make_ldxdw(0, 1, 0),
			make_exit(),
		};
		REQUIRE(verify_gpu_program(program.data(), program.size(),
					   "cuda__no_context"));
	}

	SECTION("unbounded puts helper is rejected")
	{
		const std::array<ebpf_inst, 5> program = {
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(1, 10),
			make_add64_imm(1, -8),
			make_call(501),
			make_exit(),
		};
		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__puts");
		REQUIRE(result);
		REQUIRE_FALSE(result->empty());
	}
}

TEST_CASE("Uniformity analysis classifies constants and GPU helpers",
	  "[gpu][uniformity]")
{
	SECTION("constant is UNIFORM")
	{
		const std::array<ebpf_inst, 2> program = {
			make_mov64_imm(0, 7),
			make_exit(),
		};

		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.success);
		REQUIRE(result.states.size() == program.size());
		REQUIRE(result.states[1].regs[0] == Uniformity::UNIFORM);
	}

	SECTION("thread_idx helper 505 makes R0 VARYING")
	{
		const std::array<ebpf_inst, 2> program = {
			make_call(505),
			make_exit(),
		};

		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::VARYING);
	}

	SECTION("block_idx helper 503 is UNIFORM")
	{
		const std::array<ebpf_inst, 2> program = {
			make_call(503),
			make_exit(),
		};

		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::UNIFORM);
	}

	SECTION("perf event output return is VARYING")
	{
		const std::array<ebpf_inst, 2> program = {
			make_call(25),
			make_exit(),
		};
		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.success);
		REQUIRE(result.states[1].regs[0] == Uniformity::VARYING);
	}

	SECTION("map update and delete returns are VARYING")
	{
		for (const int32_t helper : { 2, 3 }) {
			const std::array<ebpf_inst, 2> program = {
				make_call(helper),
				make_exit(),
			};
			const auto result = analyze_program_uniformity(program);
			REQUIRE(result.success);
			REQUIRE(result.states[1].regs[0] ==
				Uniformity::VARYING);
		}
	}
}

TEST_CASE("Per-thread map values are lane-varying", "[gpu][uniformity]")
{
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, make_map(1, 1502) },
	};
	const std::array<ebpf_inst, 10> program = {
		make_lddw_map(1, 1),	  {},
		make_stdw_imm(10, -8, 0), make_mov64_reg(2, 10),
		make_add64_imm(2, -8),	  make_call(1),
		make_jeq_imm(0, 0, 2),	  make_ldxdw(3, 0, 0),
		make_jeq_imm(3, 0, 0),	  make_exit(),
	};
	const auto result =
		analyze_uniformity(program.data(), program.size(), maps);
	REQUIRE(result.success);
	REQUIRE(result.states[6].regs[0] == Uniformity::UNIFORM);
	REQUIRE(result.states[6].pointers[0].offset_uniformity ==
		Uniformity::VARYING);
	REQUIRE(result.states[8].regs[3] == Uniformity::VARYING);

	const auto safety =
		check_simt_safety(program.data(), program.size(), result, maps);
	REQUIRE_FALSE(safety.passed);
}

TEST_CASE("Uniformity analysis handles extreme stack widths and offsets",
	  "[gpu][uniformity]")
{
	UniformityState state;
	state.stack_bytes.fill(Uniformity::UNIFORM);
	for (const size_t width :
	     { size_t{ 513 },
	       static_cast<size_t>(std::numeric_limits<unsigned int>::max()) }) {
		REQUIRE(query_stack_uniformity(state, -512, width) ==
			Uniformity::UNKNOWN);
	}

	const std::array<ebpf_inst, 3> program = {
		make_mov64_reg(1, 10),
		make_sub64_imm(1, std::numeric_limits<int32_t>::min()),
		make_exit(),
	};
	const auto result = analyze_program_uniformity(program);
	REQUIRE(result.success);
	REQUIRE(result.states[2].pointers[1].region == PointerRegion::STACK);
	REQUIRE_FALSE(result.states[2].pointers[1].constant_offset);

	const std::array<ebpf_inst, 4> memory_program = {
		make_mov64_reg(1, 10),
		make_add64_imm(1, std::numeric_limits<int32_t>::max()),
		make_ldxdw(0, 1, std::numeric_limits<int16_t>::max()),
		make_exit(),
	};
	const auto memory_result = analyze_program_uniformity(memory_program);
	REQUIRE(memory_result.success);
	REQUIRE(memory_result.states[3].regs[0] == Uniformity::UNKNOWN);
}

TEST_CASE("Unknown map identity is conservatively lane-varying",
	  "[gpu][uniformity]")
{
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, make_map(1, 1502) },
	};
	const std::array<ebpf_inst, 12> program = {
		make_lddw_map(1, 1),	  {},
		make_stxdw(10, 1, -16),	  make_ldxdw(1, 10, -16),
		make_stdw_imm(10, -8, 0), make_mov64_reg(2, 10),
		make_add64_imm(2, -8),	  make_call(1),
		make_jeq_imm(0, 0, 2),	  make_ldxdw(3, 0, 0),
		make_jeq_imm(3, 0, 0),	  make_exit(),
	};
	const auto result =
		analyze_uniformity(program.data(), program.size(), maps);
	REQUIRE(result.success);
	REQUIRE(result.states[8].pointers[0].offset_uniformity ==
		Uniformity::VARYING);
	REQUIRE(result.states[10].regs[3] == Uniformity::VARYING);
}

TEST_CASE("Shared map writes require lane-uniform values", "[gpu][simt]")
{
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, make_map(1, 1503) },
	};

	SECTION("direct store")
	{
		const std::array<ebpf_inst, 14> program = {
			make_lddw_map(1, 1),	  {},
			make_stdw_imm(10, -8, 0), make_mov64_reg(2, 10),
			make_add64_imm(2, -8),	  make_call(1),
			make_jeq_imm(0, 0, 6),	  make_mov64_reg(6, 0),
			make_call(511),		  make_stxdw(6, 0),
			make_ldxdw(3, 6, 0),	  make_jeq_imm(3, 0, 1),
			make_mov64_imm(0, 0),	  make_exit(),
		};
		const auto uniformity = analyze_uniformity(
			program.data(), program.size(), maps);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity, maps);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.summary().find("Shared Map Value Uniformity") !=
			std::string::npos);
	}

	SECTION("map update")
	{
		const std::array<ebpf_inst, 12> program = {
			make_call(511),
			make_stxdw(10, 0, -16),
			make_lddw_map(1, 1),
			{},
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -16),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		const auto uniformity = analyze_uniformity(
			program.data(), program.size(), maps);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity, maps);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.summary().find("Shared Map Value Uniformity") !=
			std::string::npos);
	}

	SECTION("map update after map pointer spill")
	{
		const std::array<ebpf_inst, 14> program = {
			make_call(511),
			make_stxdw(10, 0, -24),
			make_lddw_map(1, 1),
			{},
			make_stxdw(10, 1, -16),
			make_ldxdw(1, 10, -16),
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -24),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		const auto uniformity = analyze_uniformity(
			program.data(), program.size(), maps);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity, maps);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.summary().find("Shared Map Value Uniformity") !=
			std::string::npos);
	}

	SECTION("direct store after map pointer spill")
	{
		const std::array<ebpf_inst, 14> program = {
			make_call(511),		  make_stxdw(10, 0, -24),
			make_lddw_map(1, 1),	  {},
			make_stxdw(10, 1, -16),	  make_ldxdw(1, 10, -16),
			make_stdw_imm(10, -8, 0), make_mov64_reg(2, 10),
			make_add64_imm(2, -8),	  make_call(1),
			make_jeq_imm(0, 0, 2),	  make_ldxdw(3, 10, -24),
			make_stxdw(0, 3, 0),	  make_exit(),
		};
		const auto uniformity = analyze_uniformity(
			program.data(), program.size(), maps);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity, maps);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.summary().find("Shared Map Value Uniformity") !=
			std::string::npos);
	}
}

TEST_CASE("Atomic fetch results are lane-varying", "[gpu][uniformity]")
{
	SECTION("fetch updates the source register")
	{
		const std::array<ebpf_inst, 5> program = {
			make_mov64_reg(1, 10), make_add64_imm(1, -8),
			make_mov64_imm(2, 1),  make_atomic(1, 2, 0x01),
			make_exit(),
		};
		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.states[4].regs[2] == Uniformity::VARYING);
	}

	SECTION("cmpxchg updates R0")
	{
		const std::array<ebpf_inst, 6> program = {
			make_mov64_reg(1, 10),	 make_add64_imm(1, -8),
			make_mov64_imm(0, 0),	 make_mov64_imm(2, 1),
			make_atomic(1, 2, 0xf1), make_exit(),
		};
		const auto result = analyze_program_uniformity(program);
		REQUIRE(result.states[5].regs[0] == Uniformity::VARYING);
	}
}

TEST_CASE("SIMT safety enforces uniform branches and helper restrictions",
	  "[gpu][simt]")
{
	SECTION("uniform branch passes")
	{
		const std::array<ebpf_inst, 4> program = {
			make_mov64_imm(0, 1),
			make_jeq_imm(0, 1, 1),
			make_mov64_imm(0, 0),
			make_exit(),
		};

		const auto uniformity = analyze_program_uniformity(program);
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE(safety.passed);
	}

	SECTION("varying branch from thread_idx is rejected")
	{
		const std::array<ebpf_inst, 4> program = {
			make_call(505),
			make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1),
			make_exit(),
		};

		const auto uniformity = analyze_program_uniformity(program);
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
		const std::array<ebpf_inst, 2> program = {
			make_call(506),
			make_exit(),
		};

		const auto uniformity = analyze_program_uniformity(program);
		REQUIRE(uniformity.success);

		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE_FALSE(safety.errors.empty());
		REQUIRE(safety.errors[0].check_name == "Prohibited Helpers");
	}

	SECTION("lane-varying atomic address is rejected")
	{
		const std::array<ebpf_inst, 4> program = {
			make_call(505),
			make_mov64_reg(1, 0),
			make_atomic_add(1, 0),
			make_exit(),
		};

		const auto uniformity = analyze_program_uniformity(program);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
	}

	SECTION("lane-varying map key is rejected")
	{
		for (const int32_t helper : { 1, 2, 3 }) {
			const std::array<ebpf_inst, 5> program = {
				make_call(505),	      make_mov64_reg(2, 0),
				make_mov64_imm(4, 0), make_call(helper),
				make_exit(),
			};

			const auto uniformity =
				analyze_program_uniformity(program);
			const auto safety = check_simt_safety(
				program.data(), program.size(), uniformity);
			INFO(safety.summary());
			REQUIRE_FALSE(safety.passed);
		}
	}

	SECTION("lane-varying trace payload is rejected")
	{
		const std::array<ebpf_inst, 10> program = {
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(1, 10),
			make_add64_imm(1, -8),
			make_mov64_imm(2, 8),
			make_call(511),
			make_mov64_reg(3, 0),
			make_mov64_imm(4, 0),
			make_mov64_imm(5, 0),
			make_call(6),
			make_exit(),
		};

		const auto uniformity = analyze_program_uniformity(program);
		const auto safety = check_simt_safety(
			program.data(), program.size(), uniformity);
		INFO(safety.summary());
		REQUIRE_FALSE(safety.passed);
		REQUIRE(safety.summary().find(
				"Host Bridge Payload Uniformity") !=
			std::string::npos);
	}

	SECTION("lane-local pointer payloads are rejected")
	{
		const std::array<ebpf_inst, 7> map_program = {
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 2),
			make_mov64_reg(4, 10),
			make_call(2),
			make_exit(),
		};

		const auto map_uniformity =
			analyze_program_uniformity(map_program);
		const auto map_safety = check_simt_safety(
			map_program.data(), map_program.size(), map_uniformity);
		INFO(map_safety.summary());
		REQUIRE_FALSE(map_safety.passed);
		REQUIRE(map_safety.summary().find(
				"map update flags are lane-varying") !=
			std::string::npos);

		const std::array<ebpf_inst, 9> trace_program = {
			make_stdw_imm(10, -8, 0),
			make_mov64_reg(1, 10),
			make_add64_imm(1, -8),
			make_mov64_imm(2, 8),
			make_mov64_reg(3, 10),
			make_mov64_imm(4, 0),
			make_mov64_imm(5, 0),
			make_call(6),
			make_exit(),
		};

		const auto trace_uniformity =
			analyze_program_uniformity(trace_program);
		const auto trace_safety =
			check_simt_safety(trace_program.data(),
					  trace_program.size(),
					  trace_uniformity);
		INFO(trace_safety.summary());
		REQUIRE_FALSE(trace_safety.passed);
		REQUIRE(trace_safety.summary().find(
				"Host Bridge Payload Uniformity") !=
			std::string::npos);
	}
}

TEST_CASE("GPU verifier integrates SIMT phases with optional PREVAIL",
	  "[gpu][integration]")
{
	SECTION("simple safe program passes")
	{
		const std::array<ebpf_inst, 2> program = {
			make_mov64_imm(0, 0),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.value_or(""));
		REQUIRE_FALSE(result);
	}

	SECTION("empty programs fail")
	{
		const ebpf_inst sentinel{};
		const auto result =
			verify_gpu_program(&sentinel, 0, "cuda__empty");
		REQUIRE(result);
		REQUIRE(*result == "empty instruction stream");
	}

	SECTION("impossible instruction counts return an error")
	{
		const ebpf_inst sentinel{};
		const auto result =
			verify_gpu_program(&sentinel,
					   std::numeric_limits<size_t>::max(),
					   "cuda__oversized");
		REQUIRE(result);
		REQUIRE(*result ==
			"instruction count exceeds verifier capacity");
	}

	SECTION("unsafe varying branch fails")
	{
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
		INFO(result.value_or(""));
		REQUIRE(result);
		REQUIRE(result->find("Warp-Uniform Branch Conditions") !=
			std::string::npos);
	}

	SECTION("default GPU verifier runs PREVAIL for helper out-params")
	{
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
		INFO(result.value_or(""));
		REQUIRE_FALSE(result);
	}

	SECTION("PREVAIL failures reject GPU programs before SIMT")
	{
		const std::array<ebpf_inst, 3> program = {
			make_mov64_imm(1, 0),
			make_ldxdw(0, 1, 0),
			make_exit(),
		};

		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__integration");
		INFO(result.value_or(""));
		REQUIRE(result);
		REQUIRE_FALSE(result->empty());
	}

	SECTION("CPU helper registration does not widen the GPU allow-list")
	{
		set_available_helpers({ 7 });
		const std::array<ebpf_inst, 2> program = {
			make_call(7),
			make_exit(),
		};
		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__cpu_helper");
		set_available_helpers({});
		REQUIRE(result);
	}

	SECTION("ambient prototypes cannot replace fixed GPU policy")
	{
		BpftimeHelperProrotype ambient_override{};
		ambient_override.name = "ambient_override";
		ambient_override.return_type =
			bpftime::verifier::EBPF_RETURN_TYPE_INTEGER;
		for (auto &argument_type : ambient_override.argument_type) {
			argument_type =
				bpftime::verifier::EBPF_ARGUMENT_TYPE_DONTCARE;
		}
		set_available_helpers({ 1 });
		set_non_kernel_helpers({ { 1, ambient_override } });
		const std::array<ebpf_inst, 2> program = {
			make_call(1),
			make_exit(),
		};
		const auto result = verify_gpu_program(
			program.data(), program.size(), "cuda__prototype");
		set_available_helpers({});
		set_non_kernel_helpers({});
		REQUIRE(result);
	}
}
