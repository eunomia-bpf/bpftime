#include <bpftime-verifier.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ebpf_vm_isa.hpp>
#include <gpu_verifier.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

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

ebpf_inst make_jne_imm(uint8_t dst_reg, int32_t imm, int16_t off)
{
	return make_instruction(INST_CLS_JMP | INST_SRC_IMM | 0x50, dst_reg, 0,
				off, imm);
}

ebpf_inst make_jle_imm(uint8_t dst_reg, int32_t imm, int16_t off)
{
	return make_instruction(INST_CLS_JMP | INST_SRC_IMM | 0xb0, dst_reg, 0,
				off, imm);
}

ebpf_inst make_exit()
{
	return make_instruction(INST_OP_EXIT);
}

BpftimeMapDescriptor make_array_map(int fd, uint32_t type = 1503)
{
	return BpftimeMapDescriptor{
		.original_fd = fd,
		.type = type,
		.key_size = 4,
		.value_size = 16,
		.max_entries = 16,
		.inner_map_fd = static_cast<unsigned int>(-1),
	};
}

template <size_t N>
std::optional<std::string>
verify(const std::array<ebpf_inst, N> &program,
       const std::map<int, BpftimeMapDescriptor> &maps = {})
{
	return verify_gpu_program(program.data(), program.size(),
				  "cuda__revision_safety", maps);
}

void require_rejected_with(const std::optional<std::string> &result,
			   const std::string &diagnostic)
{
	INFO(result.value_or("accepted"));
	REQUIRE(result);
	REQUIRE_FALSE(result->empty());
	REQUIRE(result->find(diagnostic) != std::string::npos);
}

void require_accepted(const std::optional<std::string> &result)
{
	INFO(result.value_or("accepted"));
	REQUIRE_FALSE(result);
}

} // namespace

TEST_CASE("revision base verifier bounds pair", "[gpu][revision-safety]")
{
	SECTION("out-of-bounds stack access is rejected")
	{
		const std::array<ebpf_inst, 3> program = {
			make_stdw_imm(10, -520, 0),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_rejected_with(verify(program),
				      "Lower bound must be at least 0");
	}

	SECTION("same-width in-bounds stack access is accepted")
	{
		const std::array<ebpf_inst, 3> program = {
			make_stdw_imm(10, -8, 0),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_accepted(verify(program));
	}
}

TEST_CASE("revision base verifier loop pair", "[gpu][revision-safety]")
{
	SECTION("data-dependent backward loop without a proven bound is rejected")
	{
		const std::array<ebpf_inst, 4> program = {
			make_call(511),
			make_jle_imm(0, 3, -2),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_rejected_with(verify(program),
				      "Could not prove termination");
	}

	SECTION("constant-bounded backward loop is accepted")
	{
		const std::array<ebpf_inst, 5> program = {
			make_mov64_imm(1, 1),
			make_sub64_imm(1, 1),
			make_jne_imm(1, 0, -2),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_accepted(verify(program));
	}
}

TEST_CASE("revision SIMT branch pair", "[gpu][revision-safety]")
{
	SECTION("lane-derived predicate is rejected")
	{
		const std::array<ebpf_inst, 4> program = {
			make_call(511),
			make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1),
			make_exit(),
		};
		require_rejected_with(verify(program),
				      "Warp-Uniform Branch Conditions");
	}

	SECTION("warp-uniform block predicate is accepted")
	{
		const std::array<ebpf_inst, 4> program = {
			make_call(510),
			make_jeq_imm(0, 0, 1),
			make_mov64_imm(0, 1),
			make_exit(),
		};
		require_accepted(verify(program));
	}
}

TEST_CASE("revision SIMT map side-effect pairs", "[gpu][revision-safety]")
{
	const std::map<int, BpftimeMapDescriptor> maps = {
		{ 1, make_array_map(1) },
	};

	SECTION("lane-derived map key is rejected")
	{
		const std::array<ebpf_inst, 12> program = {
			make_call(511),
			make_stxdw(10, 0, -8),
			make_stdw_imm(10, -16, 7),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -16),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		require_rejected_with(verify(program, maps),
				      "Map Helper Key Uniformity");
	}

	SECTION("warp-uniform map key is accepted")
	{
		const std::array<ebpf_inst, 12> program = {
			make_call(510),
			make_stxdw(10, 0, -8),
			make_stdw_imm(10, -16, 7),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -16),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		require_accepted(verify(program, maps));
	}

	SECTION("lane-derived shared-map value is rejected")
	{
		const std::array<ebpf_inst, 12> program = {
			make_stdw_imm(10, -8, 0),
			make_call(511),
			make_stxdw(10, 0, -16),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -16),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		require_rejected_with(verify(program, maps),
				      "Shared Map Value Uniformity");
	}

	SECTION("warp-uniform shared-map value is accepted")
	{
		const std::array<ebpf_inst, 12> program = {
			make_stdw_imm(10, -8, 0),
			make_call(510),
			make_stxdw(10, 0, -16),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_mov64_reg(3, 10),
			make_add64_imm(3, -16),
			make_mov64_imm(4, 0),
			make_call(2),
			make_exit(),
		};
		require_accepted(verify(program, maps));
	}
}

TEST_CASE("revision SIMT atomic and helper pairs", "[gpu][revision-safety]")
{
	SECTION("lane-varying atomic target is rejected")
	{
		const std::map<int, BpftimeMapDescriptor> maps = {
			{ 1, make_array_map(1, 1502) },
		};
		const std::array<ebpf_inst, 12> program = {
			make_stdw_imm(10, -8, 0),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_call(1),
			make_jeq_imm(0, 0, 3),
			make_mov64_reg(1, 0),
			make_mov64_imm(2, 1),
			make_atomic(1, 2, 0),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_rejected_with(verify(program, maps),
				      "Atomic Operations on Uniform Addresses");
	}

	SECTION("warp-uniform atomic target is accepted")
	{
		const std::map<int, BpftimeMapDescriptor> maps = {
			{ 1, make_array_map(1) },
		};
		const std::array<ebpf_inst, 12> program = {
			make_stdw_imm(10, -8, 0),
			make_lddw_map(1, 1),
			{},
			make_mov64_reg(2, 10),
			make_add64_imm(2, -8),
			make_call(1),
			make_jeq_imm(0, 0, 3),
			make_mov64_reg(1, 0),
			make_mov64_imm(2, 1),
			make_atomic(1, 2, 0),
			make_mov64_imm(0, 0),
			make_exit(),
		};
		require_accepted(verify(program, maps));
	}

	SECTION("prohibited helper is rejected")
	{
		const std::array<ebpf_inst, 2> program = {
			make_call(506),
			make_exit(),
		};
		require_rejected_with(verify(program), "Prohibited Helpers");
	}

	SECTION("allowed helper is accepted")
	{
		const std::array<ebpf_inst, 2> program = {
			make_call(510),
			make_exit(),
		};
		require_accepted(verify(program));
	}
}
