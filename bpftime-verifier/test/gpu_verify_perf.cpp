#include <ebpf_inst.h>
#include <gpu_verifier.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{

constexpr std::size_t MIN_PROGRAM_SIZE = 2;

ebpf_inst make_instruction(uint8_t code, uint8_t dst_reg, uint8_t src_reg,
			   int16_t off, int32_t imm)
{
	ebpf_inst instruction {};
	instruction.code = code;
	instruction.dst_reg = dst_reg;
	instruction.src_reg = src_reg;
	instruction.off = off;
	instruction.imm = imm;
	return instruction;
}

std::vector<ebpf_inst> build_program(std::size_t instruction_count)
{
	if (instruction_count < MIN_PROGRAM_SIZE) {
		throw std::invalid_argument(
			"program must contain at least 2 instructions");
	}

	std::vector<ebpf_inst> program;
	program.reserve(instruction_count);

	program.push_back(
		make_instruction(EBPF_OP_MOV64_IMM, 0, 0, 0, 0));

	for (std::size_t i = 1; i + 1 < instruction_count; ++i) {
		switch ((i - 1) % 3) {
		case 0:
			program.push_back(make_instruction(
				EBPF_OP_MOV64_REG, 1, 0, 0, 0));
			break;
		case 1:
			program.push_back(make_instruction(
				EBPF_OP_ADD64_IMM, 1, 0, 0, 1));
			break;
		default:
			program.push_back(make_instruction(
				EBPF_OP_MOV64_REG, 0, 1, 0, 0));
			break;
		}
	}

	program.push_back(make_instruction(EBPF_OP_EXIT, 0, 0, 0, 0));
	return program;
}

std::size_t parse_program_size(int argc, char **argv)
{
	if (argc == 2) {
		return std::stoull(argv[1]);
	}

	if (argc == 3 && std::string_view(argv[1]) == "--size") {
		return std::stoull(argv[2]);
	}

	throw std::invalid_argument(
		"usage: bpftime_gpu_verify_perf <size> or bpftime_gpu_verify_perf --size <size>");
}

} // namespace

int main(int argc, char **argv)
{
	try {
		const std::size_t program_size = parse_program_size(argc, argv);
		const auto program = build_program(program_size);
		bpftime::verifier::gpu::GpuVerifierConfig config;
		config.budget = bpftime::verifier::gpu::GpuResourceBudget{
			.max_instructions = static_cast<uint32_t>(
				std::max<std::size_t>(8192, program_size * 4)),
			.max_helper_calls = 512,
			.max_memory_ops = 1024,
			.max_map_lookups = 128,
			.max_map_updates = 128,
		};

		const auto result = bpftime::verifier::gpu::verify_gpu_program(
			program.data(), program.size(), "kprobe/cuda__bench", config);

		if (!result.passed) {
			std::cerr << "GPU verifier rejected synthetic program: "
				  << result.error_message << '\n';
			return EXIT_FAILURE;
		}

		std::cout << "{"
			  << "\"program_size\":" << program.size() << ","
			  << "\"total_time_us\":" << std::fixed
			  << std::setprecision(3) << result.total_time_us << ","
			  << "\"prevail_time_us\":" << result.prevail_time_us << ","
			  << "\"simt_time_us\":" << result.simt_time_us << "}"
			  << '\n';
		return EXIT_SUCCESS;
	} catch (const std::exception &ex) {
		std::cerr << ex.what() << '\n';
		return EXIT_FAILURE;
	}
}
