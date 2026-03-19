#include <asm_files.hpp>
#include <bpftime-verifier.hpp>
#include <gelf.h>
#include <gpu_platform.hpp>
#include <gpu_verifier.hpp>
#include <libelf.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;

namespace
{

constexpr uint8_t OP_MOV64_IMM = 0xb7;
constexpr uint8_t OP_DIV64_REG = 0x3f;
constexpr uint8_t OP_LDXDW = 0x79;
constexpr uint8_t OP_JA = 0x05;
constexpr uint8_t OP_EXIT = 0x95;

constexpr std::array<int32_t, 18> AVAILABLE_HELPERS = {
	1,   2,	  3,   5,   6,	 14,  25,  501, 502,
	503, 504, 505, 506, 507, 508, 509, 510, 511,
};

enum class EvalMode {
	GENERIC,
	GPU_SIMT,
	GPU_PREVAIL,
};

struct LoadedProgram {
	std::filesystem::path object_path;
	std::string section_name;
	std::vector<ebpf_inst> instructions;
	std::map<int, BpftimeMapDescriptor> map_descriptors;
};

struct VerificationRow {
	std::string section_name;
	std::string engine;
	bool passed = false;
	std::string error_message;
	double total_time_us = 0.0;
	double prevail_time_us = 0.0;
	double simt_time_us = 0.0;
	uint32_t instruction_count = 0;
	uint32_t helper_call_count = 0;
	uint32_t memory_op_count = 0;
	uint32_t map_lookup_count = 0;
	uint32_t map_update_count = 0;
	uint32_t varying_branch_count = 0;
	uint32_t prohibited_helper_count = 0;
};

class ScopedFd {
    public:
	explicit ScopedFd(const std::filesystem::path &path)
	{
		fd_ = open(path.c_str(), O_RDONLY);
		if (fd_ < 0) {
			throw std::system_error(errno, std::generic_category(),
						"open " + path.string());
		}
	}

	~ScopedFd()
	{
		if (fd_ >= 0) {
			close(fd_);
		}
	}

	ScopedFd(const ScopedFd &) = delete;
	ScopedFd &operator=(const ScopedFd &) = delete;

	int get() const
	{
		return fd_;
	}

    private:
	int fd_ = -1;
};

class ScopedElf {
    public:
	explicit ScopedElf(int fd)
	{
		handle_ = elf_begin(fd, ELF_C_READ, nullptr);
		if (handle_ == nullptr) {
			throw std::runtime_error(
				std::string("elf_begin failed: ") +
				elf_errmsg(-1));
		}
	}

	~ScopedElf()
	{
		if (handle_ != nullptr) {
			elf_end(handle_);
		}
	}

	ScopedElf(const ScopedElf &) = delete;
	ScopedElf &operator=(const ScopedElf &) = delete;

	Elf *get() const
	{
		return handle_;
	}

    private:
	Elf *handle_ = nullptr;
};

void ensure_libelf_ready()
{
	static const bool initialized = [] {
		return elf_version(EV_CURRENT) != EV_NONE;
	}();
	if (!initialized) {
		throw std::runtime_error(std::string("elf_version failed: ") +
					 elf_errmsg(-1));
	}
}

bool is_program_section(const GElf_Shdr &header, const char *name)
{
	return (header.sh_flags & SHF_EXECINSTR) != 0 && header.sh_size > 0 &&
	       name != nullptr && name[0] != '.';
}

std::vector<std::string>
list_program_sections(const std::filesystem::path &object_path)
{
	ensure_libelf_ready();
	ScopedFd fd(object_path);
	ScopedElf elf(fd.get());

	GElf_Ehdr ehdr{};
	if (gelf_getehdr(elf.get(), &ehdr) == nullptr) {
		throw std::runtime_error(std::string("gelf_getehdr failed: ") +
					 elf_errmsg(-1));
	}

	std::vector<std::string> sections;
	Elf_Scn *section = nullptr;
	while ((section = elf_nextscn(elf.get(), section)) != nullptr) {
		GElf_Shdr shdr{};
		if (gelf_getshdr(section, &shdr) == nullptr) {
			throw std::runtime_error(
				std::string("gelf_getshdr failed: ") +
				elf_errmsg(-1));
		}

		const char *section_name =
			elf_strptr(elf.get(), ehdr.e_shstrndx, shdr.sh_name);
		if (is_program_section(shdr, section_name)) {
			sections.emplace_back(section_name);
		}
	}

	return sections;
}

std::map<int, BpftimeMapDescriptor>
convert_map_descriptors(const raw_program &program)
{
	std::map<int, BpftimeMapDescriptor> descriptors;
	for (const auto &map : program.info.map_descriptors) {
		descriptors.emplace(map.original_fd,
				    BpftimeMapDescriptor{
					    .original_fd = map.original_fd,
					    .type = map.type,
					    .key_size = map.key_size,
					    .value_size = map.value_size,
					    .max_entries = map.max_entries,
					    .inner_map_fd = map.inner_map_fd,
				    });
	}
	return descriptors;
}

LoadedProgram load_program(const std::filesystem::path &object_path,
			   const std::string &section_name)
{
	auto programs = read_elf(object_path.string(), section_name, nullptr,
				 &bpftime::gpu_platform_spec);
	if (programs.size() != 1) {
		throw std::runtime_error(
			"expected exactly one relocated program in " +
			object_path.string() + ":" + section_name);
	}

	const auto &program = programs.front();
	return LoadedProgram{
		.object_path = object_path,
		.section_name = program.section,
		.instructions = program.prog,
		.map_descriptors = convert_map_descriptors(program),
	};
}

std::vector<LoadedProgram>
load_programs(const std::filesystem::path &object_path)
{
	const auto section_names = list_program_sections(object_path);
	if (section_names.empty()) {
		throw std::runtime_error(
			"no executable BPF sections found in " +
			object_path.string());
	}

	std::vector<LoadedProgram> programs;
	programs.reserve(section_names.size());
	for (const auto &section_name : section_names) {
		programs.push_back(load_program(object_path, section_name));
	}
	return programs;
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

std::map<int32_t, BpftimeHelperProrotype> make_gpu_helper_prototypes()
{
	using bpftime::verifier::EBPF_ARGUMENT_TYPE_ANYTHING;
	using bpftime::verifier::EBPF_ARGUMENT_TYPE_DONTCARE;
	using bpftime::verifier::EBPF_RETURN_TYPE_INTEGER;
	using bpftime::verifier::EBPF_RETURN_TYPE_INTEGER_OR_NO_RETURN_IF_SUCCEED;

	return {
		{ 501,
		  make_helper_prototype("ebpf_puts", EBPF_RETURN_TYPE_INTEGER,
					{ EBPF_ARGUMENT_TYPE_ANYTHING }) },
		{ 502, make_helper_prototype("bpf_get_globaltimer",
					     EBPF_RETURN_TYPE_INTEGER, {}) },
		{ 503, make_helper_prototype(
			       "bpf_get_block_idx", EBPF_RETURN_TYPE_INTEGER,
			       {
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
			       }) },
		{ 504, make_helper_prototype(
			       "bpf_get_block_dim", EBPF_RETURN_TYPE_INTEGER,
			       {
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
			       }) },
		{ 505, make_helper_prototype(
			       "bpf_get_thread_idx", EBPF_RETURN_TYPE_INTEGER,
			       {
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
			       }) },
		{ 506, make_helper_prototype("bpf_gpu_membar",
					     EBPF_RETURN_TYPE_INTEGER, {}) },
		{ 507, make_helper_prototype(
			       "bpf_cuda_exit",
			       EBPF_RETURN_TYPE_INTEGER_OR_NO_RETURN_IF_SUCCEED,
			       {}) },
		{ 508, make_helper_prototype(
			       "bpf_get_grid_dim", EBPF_RETURN_TYPE_INTEGER,
			       {
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
				       EBPF_ARGUMENT_TYPE_ANYTHING,
			       }) },
		{ 509, make_helper_prototype("bpf_get_sm_id",
					     EBPF_RETURN_TYPE_INTEGER, {}) },
		{ 510, make_helper_prototype("bpf_get_warp_id",
					     EBPF_RETURN_TYPE_INTEGER, {}) },
		{ 511, make_helper_prototype("bpf_get_lane_id",
					     EBPF_RETURN_TYPE_INTEGER, {}) },
	};
}

void reset_verifier_state(const std::map<int, BpftimeMapDescriptor> &maps)
{
	set_available_helpers(std::vector<int32_t>(AVAILABLE_HELPERS.begin(),
						   AVAILABLE_HELPERS.end()));
	set_non_kernel_helpers(make_gpu_helper_prototypes());
	set_map_descriptors(maps);
}

ebpf_inst make_instruction(uint8_t code, uint8_t dst_reg, uint8_t src_reg,
			   int16_t off, int32_t imm)
{
	ebpf_inst instruction{};
	instruction.opcode = code;
	instruction.dst = dst_reg;
	instruction.src = src_reg;
	instruction.offset = off;
	instruction.imm = imm;
	return instruction;
}

std::vector<ebpf_inst> build_builtin_pattern(const std::string &name)
{
	if (name == "null_deref") {
		return {
			make_instruction(OP_MOV64_IMM, 1, 0, 0, 0),
			make_instruction(OP_LDXDW, 0, 1, 0, 0),
			make_instruction(OP_EXIT, 0, 0, 0, 0),
		};
	}
	if (name == "division_by_zero") {
		return {
			make_instruction(OP_MOV64_IMM, 0, 0, 0, 1),
			make_instruction(OP_MOV64_IMM, 1, 0, 0, 0),
			make_instruction(OP_DIV64_REG, 0, 1, 0, 0),
			make_instruction(OP_EXIT, 0, 0, 0, 0),
		};
	}
	if (name == "resource_exceeded") {
		return {
			make_instruction(OP_JA, 0, 0, -1, 0),
			make_instruction(OP_EXIT, 0, 0, 0, 0),
		};
	}

	throw std::invalid_argument("unknown builtin pattern: " + name);
}

std::string json_escape(std::string_view value)
{
	std::string escaped;
	escaped.reserve(value.size());
	for (const char ch : value) {
		switch (ch) {
		case '\\':
			escaped += "\\\\";
			break;
		case '"':
			escaped += "\\\"";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped += ch;
			break;
		}
	}
	return escaped;
}

EvalMode parse_mode(const std::string &value)
{
	if (value == "generic") {
		return EvalMode::GENERIC;
	}
	if (value == "gpu-simt") {
		return EvalMode::GPU_SIMT;
	}
	if (value == "gpu-prevail") {
		return EvalMode::GPU_PREVAIL;
	}
	throw std::invalid_argument("unknown mode: " + value);
}

VerificationRow verify_gpu_section(const LoadedProgram &program, EvalMode mode);

VerificationRow verify_generic_section(const LoadedProgram &program)
{
	reset_verifier_state(program.map_descriptors);
	VerificationRow row;
	row.section_name = program.section_name;
	if (program.section_name.starts_with("kprobe/") ||
	    program.section_name.starts_with("kretprobe/")) {
		const auto gpu_row =
			verify_gpu_section(program, EvalMode::GPU_SIMT);
		row = gpu_row;
		row.engine = "gpu-simt";
		return row;
	}
	const auto prevail_row =
		verify_gpu_section(program, EvalMode::GPU_PREVAIL);
	row = prevail_row;
	row.engine = "prevail";
	return row;
}

VerificationRow verify_gpu_section(const LoadedProgram &program, EvalMode mode)
{
	reset_verifier_state(program.map_descriptors);

	GpuVerifierConfig config;
	config.mode = (mode == EvalMode::GPU_PREVAIL) ?
			      GpuVerificationMode::PREVAIL_ONLY :
			      GpuVerificationMode::SIMT_AWARE;
	config.skip_prevail = false;
	config.map_descriptors = program.map_descriptors;

	const auto result = verify_gpu_program(program.instructions.data(),
					       program.instructions.size(),
					       program.section_name, config);

	VerificationRow row;
	row.section_name = program.section_name;
	row.engine =
		(mode == EvalMode::GPU_PREVAIL) ? "gpu-prevail" : "gpu-simt";
	row.passed = result.passed;
	row.error_message = result.error_message;
	row.total_time_us = result.total_time_us;
	row.prevail_time_us = result.prevail_time_us;
	row.simt_time_us = result.simt_time_us;
	row.instruction_count = result.instruction_count;
	row.helper_call_count = result.helper_call_count;
	row.memory_op_count = result.memory_op_count;
	row.map_lookup_count = result.map_lookup_count;
	row.map_update_count = result.map_update_count;
	row.varying_branch_count = result.varying_branch_count;
	row.prohibited_helper_count = result.prohibited_helper_count;
	return row;
}

VerificationRow verify_builtin_pattern(const std::string &pattern,
				       EvalMode mode)
{
	if (mode == EvalMode::GENERIC) {
		throw std::invalid_argument(
			"builtin patterns support only gpu-simt or gpu-prevail modes");
	}

	const LoadedProgram program{
		.object_path = pattern,
		.section_name = "kprobe/cuda__comparison",
		.instructions = build_builtin_pattern(pattern),
		.map_descriptors = {},
	};
	return verify_gpu_section(program, mode);
}

void print_row_json(const VerificationRow &row)
{
	std::cout
		<< "{"
		<< "\"section_name\":\"" << json_escape(row.section_name)
		<< "\","
		<< "\"engine\":\"" << json_escape(row.engine) << "\","
		<< "\"passed\":" << (row.passed ? "true" : "false") << ","
		<< "\"error_message\":\"" << json_escape(row.error_message)
		<< "\","
		<< "\"total_time_us\":" << std::fixed << std::setprecision(3)
		<< row.total_time_us << ","
		<< "\"prevail_time_us\":" << row.prevail_time_us << ","
		<< "\"simt_time_us\":" << row.simt_time_us << ","
		<< "\"instruction_count\":" << row.instruction_count << ","
		<< "\"helper_call_count\":" << row.helper_call_count << ","
		<< "\"memory_op_count\":" << row.memory_op_count << ","
		<< "\"map_lookup_count\":" << row.map_lookup_count << ","
		<< "\"map_update_count\":" << row.map_update_count << ","
		<< "\"varying_branch_count\":" << row.varying_branch_count
		<< ","
		<< "\"prohibited_helper_count\":" << row.prohibited_helper_count
		<< "}";
}

void print_usage(const char *argv0)
{
	std::cerr
		<< "usage:\n"
		<< "  " << argv0
		<< " verify-object <object-path> [--mode generic|gpu-simt|gpu-prevail]\n"
		<< "  " << argv0
		<< " verify-builtin <null_deref|division_by_zero|resource_exceeded>"
		<< " [--mode gpu-simt|gpu-prevail]\n";
}

} // namespace

int main(int argc, char **argv)
{
	try {
		if (argc < 3) {
			print_usage(argv[0]);
			return 1;
		}

		const std::string command = argv[1];
		const std::string target = argv[2];
		EvalMode mode = (command == "verify-builtin") ?
					EvalMode::GPU_SIMT :
					EvalMode::GENERIC;

		for (int i = 3; i < argc; ++i) {
			if (std::string_view(argv[i]) == "--mode") {
				if (i + 1 >= argc) {
					throw std::invalid_argument(
						"--mode requires an argument");
				}
				mode = parse_mode(argv[++i]);
				continue;
			}
			throw std::invalid_argument("unknown argument: " +
						    std::string(argv[i]));
		}

		std::cout << "{";
		if (command == "verify-object") {
			const auto programs = load_programs(target);
			std::cout
				<< "\"kind\":\"object\","
				<< "\"target\":\"" << json_escape(target)
				<< "\","
				<< "\"mode\":\""
				<< json_escape(
					   mode == EvalMode::GENERIC ?
						   "generic" :
						   (mode == EvalMode::GPU_SIMT ?
							    "gpu-simt" :
							    "gpu-prevail"))
				<< "\","
				<< "\"programs\":[";
			for (size_t i = 0; i < programs.size(); ++i) {
				VerificationRow row =
					(mode == EvalMode::GENERIC) ?
						verify_generic_section(
							programs[i]) :
						verify_gpu_section(programs[i],
								   mode);
				if (i > 0) {
					std::cout << ",";
				}
				print_row_json(row);
			}
			std::cout << "]";
		} else if (command == "verify-builtin") {
			const auto row = verify_builtin_pattern(target, mode);
			std::cout << "\"kind\":\"builtin\","
				  << "\"target\":\"" << json_escape(target)
				  << "\","
				  << "\"mode\":\""
				  << json_escape(mode == EvalMode::GPU_SIMT ?
							 "gpu-simt" :
							 "gpu-prevail")
				  << "\","
				  << "\"programs\":[";
			print_row_json(row);
			std::cout << "]";
		} else {
			throw std::invalid_argument("unknown command: " +
						    command);
		}

		std::cout << "}\n";
		return 0;
	} catch (const std::exception &ex) {
		std::cerr << ex.what() << '\n';
		return 1;
	}
}
