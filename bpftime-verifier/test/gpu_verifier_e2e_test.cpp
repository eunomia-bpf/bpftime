#include <asm_files.hpp>
#include <bpftime-verifier.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <gelf.h>
#include <gpu_platform.hpp>
#include <gpu_verifier.hpp>
#include <libelf.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

#ifndef BPFTIME_GPU_BPF_OBJECT_DIR
#error "BPFTIME_GPU_BPF_OBJECT_DIR must be defined"
#endif

#define BPFTIME_STRINGIZE_DETAIL(x) #x
#define BPFTIME_STRINGIZE(x) BPFTIME_STRINGIZE_DETAIL(x)

using namespace bpftime;
using namespace bpftime::verifier;
using namespace bpftime::verifier::gpu;
using Catch::Matchers::ContainsSubstring;

namespace
{

constexpr std::array<int32_t, 18> AVAILABLE_HELPERS = {
	1,   2,   3,   5,   6,   14,  25,  501, 502,
	503, 504, 505, 506, 507, 508, 509, 510, 511,
};

struct LoadedGpuProgram {
	std::filesystem::path object_path;
	std::string section_name;
	std::vector<uint8_t> section_bytes;
	std::vector<ebpf_inst> instructions;
	std::map<int, BpftimeMapDescriptor> map_descriptors;
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
			throw std::runtime_error(std::string("elf_begin failed: ") +
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

	GElf_Ehdr ehdr {};
	if (gelf_getehdr(elf.get(), &ehdr) == nullptr) {
		throw std::runtime_error(
			std::string("gelf_getehdr failed: ") +
			elf_errmsg(-1));
	}

	std::vector<std::string> sections;
	Elf_Scn *section = nullptr;
	while ((section = elf_nextscn(elf.get(), section)) != nullptr) {
		GElf_Shdr shdr {};
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

std::vector<uint8_t> read_program_section_bytes(
	const std::filesystem::path &object_path, const std::string &section_name)
{
	ensure_libelf_ready();
	ScopedFd fd(object_path);
	ScopedElf elf(fd.get());

	GElf_Ehdr ehdr {};
	if (gelf_getehdr(elf.get(), &ehdr) == nullptr) {
		throw std::runtime_error(
			std::string("gelf_getehdr failed: ") +
			elf_errmsg(-1));
	}

	Elf_Scn *section = nullptr;
	while ((section = elf_nextscn(elf.get(), section)) != nullptr) {
		GElf_Shdr shdr {};
		if (gelf_getshdr(section, &shdr) == nullptr) {
			throw std::runtime_error(
				std::string("gelf_getshdr failed: ") +
				elf_errmsg(-1));
		}

		const char *current_name =
			elf_strptr(elf.get(), ehdr.e_shstrndx, shdr.sh_name);
		if (!is_program_section(shdr, current_name) ||
		    section_name != current_name) {
			continue;
		}

		std::vector<uint8_t> bytes;
		Elf_Data *data = nullptr;
		while ((data = elf_getdata(section, data)) != nullptr) {
			if (data->d_buf == nullptr || data->d_size == 0) {
				continue;
			}
			const auto *begin =
				static_cast<const uint8_t *>(data->d_buf);
			bytes.insert(bytes.end(), begin, begin + data->d_size);
		}
		return bytes;
	}

	throw std::runtime_error("missing program section " + section_name +
				 " in " + object_path.string());
}

std::map<int, BpftimeMapDescriptor>
convert_map_descriptors(const raw_program &program)
{
	std::map<int, BpftimeMapDescriptor> descriptors;
	for (const auto &map : program.info.map_descriptors) {
		descriptors.emplace(
			map.original_fd,
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

LoadedGpuProgram load_gpu_program(const std::filesystem::path &object_path,
				  const std::string &section_name)
{
	auto section_bytes =
		read_program_section_bytes(object_path, section_name);
	if ((section_bytes.size() % sizeof(ebpf_inst)) != 0) {
		throw std::runtime_error("section size is not instruction-aligned: " +
					 object_path.string() + ":" + section_name);
	}

	auto programs =
		read_elf(object_path.string(), section_name, nullptr,
			 &bpftime::gpu_platform_spec);
	if (programs.size() != 1) {
		throw std::runtime_error("expected exactly one relocated program in " +
					 object_path.string() + ":" + section_name);
	}

	const auto &program = programs.front();
	if (section_bytes.size() !=
	    program.prog.size() * sizeof(ebpf_inst)) {
		throw std::runtime_error(
			"raw section bytes and relocated instruction count disagree for " +
			object_path.string() + ":" + section_name);
	}

	return LoadedGpuProgram{
		.object_path = object_path,
		.section_name = program.section,
		.section_bytes = std::move(section_bytes),
		.instructions = program.prog,
		.map_descriptors = convert_map_descriptors(program),
	};
}

std::vector<LoadedGpuProgram>
load_gpu_programs(const std::filesystem::path &object_path)
{
	const auto section_names = list_program_sections(object_path);
	if (section_names.empty()) {
		throw std::runtime_error("no executable BPF sections found in " +
					 object_path.string());
	}

	std::vector<LoadedGpuProgram> programs;
	programs.reserve(section_names.size());
	for (const auto &section_name : section_names) {
		programs.push_back(load_gpu_program(object_path, section_name));
	}
	return programs;
}

BpftimeHelperProrotype make_helper_prototype(
	const char *name, bpftime_return_type_t return_type,
	std::initializer_list<bpftime_argument_type_t> argument_types)
{
	BpftimeHelperProrotype prototype {};
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
	return {
		{ 501,
		  make_helper_prototype("ebpf_puts",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{ bpftime::verifier::
						  EBPF_ARGUMENT_TYPE_ANYTHING }) },
		{ 502,
		  make_helper_prototype("bpf_get_globaltimer",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{}) },
		{ 503,
		  make_helper_prototype("bpf_get_block_idx",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
					}) },
		{ 504,
		  make_helper_prototype("bpf_get_block_dim",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
					}) },
		{ 505,
		  make_helper_prototype("bpf_get_thread_idx",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
					}) },
		{ 506,
		  make_helper_prototype("bpf_gpu_membar",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{}) },
		{ 507,
		  make_helper_prototype(
			  "bpf_cuda_exit",
			  bpftime::verifier::
				  EBPF_RETURN_TYPE_INTEGER_OR_NO_RETURN_IF_SUCCEED,
			  {}) },
		{ 508,
		  make_helper_prototype("bpf_get_grid_dim",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
						bpftime::verifier::
							EBPF_ARGUMENT_TYPE_ANYTHING,
					}) },
		{ 509,
		  make_helper_prototype("bpf_get_sm_id",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{}) },
		{ 510,
		  make_helper_prototype("bpf_get_warp_id",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{}) },
		{ 511,
		  make_helper_prototype("bpf_get_lane_id",
					bpftime::verifier::
						EBPF_RETURN_TYPE_INTEGER,
					{}) },
	};
}

void reset_verifier_state(const std::map<int, BpftimeMapDescriptor> &maps)
{
	set_available_helpers(std::vector<int32_t>(AVAILABLE_HELPERS.begin(),
						 AVAILABLE_HELPERS.end()));
	set_non_kernel_helpers(make_gpu_helper_prototypes());
	set_map_descriptors(maps);
}

GpuVerifyResult verify_loaded_program(const LoadedGpuProgram &program)
{
	reset_verifier_state(program.map_descriptors);
	return verify_gpu_program(program.instructions.data(),
				  program.instructions.size(),
				  program.section_name);
}

std::filesystem::path generated_object_path(const std::string &file_name)
{
	return std::filesystem::path(
		       BPFTIME_STRINGIZE(BPFTIME_GPU_BPF_OBJECT_DIR)) /
	       file_name;
}

} // namespace

TEST_CASE("Compiled unsafe GPU programs are rejected", "[gpu][e2e][unsafe]")
{
	struct ExpectedFailure {
		const char *object_name;
		const char *error_snippet;
	};

	const std::array<ExpectedFailure, 5> expected_failures = {
		ExpectedFailure{ "varying_branch.bpf.o",
				 "Warp-Uniform Branch Conditions" },
		ExpectedFailure{ "prohibited_helper.bpf.o",
				 "Prohibited Helpers" },
		ExpectedFailure{ "varying_atomic.bpf.o",
				 "Atomic Operations on Uniform Addresses" },
		ExpectedFailure{ "varying_map_key.bpf.o",
				 "Map Update Key Uniformity" },
		ExpectedFailure{ "resource_exceeded.bpf.o",
				 "resource budget exceeded" },
	};

	for (const auto &expected : expected_failures) {
		const auto object_path =
			generated_object_path(expected.object_name);
		CAPTURE(object_path.string());

		const auto programs = load_gpu_programs(object_path);
		REQUIRE(programs.size() == 1);

		const auto result = verify_loaded_program(programs.front());
		INFO(result.error_message);
		REQUIRE_FALSE(result.passed);
		REQUIRE_THAT(result.error_message,
			     ContainsSubstring(expected.error_snippet));

		if (std::string_view(expected.object_name) ==
		    "varying_branch.bpf.o") {
			REQUIRE(result.varying_branch_count >= 1);
		}
		if (std::string_view(expected.object_name) ==
		    "prohibited_helper.bpf.o") {
			REQUIRE(result.prohibited_helper_count == 1);
		}
		if (std::string_view(expected.object_name) ==
		    "resource_exceeded.bpf.o") {
			REQUIRE(result.helper_call_count > 64);
		}
	}
}

TEST_CASE("Compiled safe GPU programs pass", "[gpu][e2e][safe]")
{
	const std::array<const char *, 2> safe_objects = {
		"safe_counter.bpf.o",
		"safe_block_idx_branch.bpf.o",
	};

	for (const auto *object_name : safe_objects) {
		const auto object_path = generated_object_path(object_name);
		CAPTURE(object_path.string());

		const auto programs = load_gpu_programs(object_path);
		REQUIRE(programs.size() == 1);

		const auto result = verify_loaded_program(programs.front());
		INFO(result.error_message);
		REQUIRE(result.passed);
	}
}

TEST_CASE("Real GPU example programs pass without false positives",
	  "[gpu][e2e][examples]")
{
	const std::array<const char *, 1> example_objects = {
		"cuda_probe.bpf.o",
	};

	for (const auto *object_name : example_objects) {
		const auto object_path = generated_object_path(object_name);
		const auto programs = load_gpu_programs(object_path);
		REQUIRE_FALSE(programs.empty());

		for (const auto &program : programs) {
			CAPTURE(object_path.string());
			CAPTURE(program.section_name);

			const auto result = verify_loaded_program(program);
			INFO(result.error_message);
			REQUIRE(result.passed);
		}
	}
}

TEST_CASE("Compiled example objects stay ELF-loadable", "[gpu][e2e][elf]")
{
	const auto programs =
		load_gpu_programs(generated_object_path("gpu_shared_map.bpf.o"));
	REQUIRE(programs.size() == 2);

	for (const auto &program : programs) {
		REQUIRE_FALSE(program.instructions.empty());
		REQUIRE(program.section_bytes.size() ==
			program.instructions.size() * sizeof(ebpf_inst));
	}
}
