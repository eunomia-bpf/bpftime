#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_verifier.hpp"
#include "sass_aot.hpp"
#include "sass_aot_test_config.hpp"

namespace
{

using namespace bpftime::attach::sass_aot;

constexpr const char *kSpikeSection = "cuda__/sass_aot";
constexpr const char *kFuncName = "sass_aot_probe";

void require(bool condition, const std::string &message)
{
	if (!condition)
		throw std::runtime_error(message);
}

// Same instruction layout as ebpf_inst / PREVAIL's ebpf_vm_isa.
uint64_t make_word(uint8_t opcode, uint8_t dst = 0, uint8_t src = 0,
		   int16_t offset = 0, int32_t imm = 0)
{
	return static_cast<uint64_t>(opcode) |
	       (static_cast<uint64_t>(dst) << 8) |
	       (static_cast<uint64_t>(src) << 12) |
	       (static_cast<uint64_t>(static_cast<uint16_t>(offset)) << 16) |
	       (static_cast<uint64_t>(static_cast<uint32_t>(imm)) << 32);
}

std::vector<uint64_t> make_lane_varying_branch_program()
{
	const auto mov64_reg = [](uint8_t dst, uint8_t src) {
		return make_word(0xbf, dst, src);
	};
	const auto add64_imm = [](uint8_t dst, int32_t imm) {
		return make_word(0x07, dst, 0, 0, imm);
	};
	const auto call = [](int32_t helper_id) {
		return make_word(0x85, 0, 0, 0, helper_id);
	};
	const auto ldxdw = [](uint8_t dst, uint8_t src, int16_t off) {
		return make_word(0x79, dst, src, off);
	};
	const auto jeq_imm = [](uint8_t dst, int32_t imm, int16_t off) {
		return make_word(0x15, dst, 0, off, imm);
	};
	const auto mov64_imm = [](uint8_t dst, int32_t imm) {
		return make_word(0xb7, dst, 0, 0, imm);
	};

	return { mov64_reg(1, 10),  add64_imm(1, -8), mov64_reg(2, 10),
		 add64_imm(2, -16), mov64_reg(3, 10), add64_imm(3, -24),
		 call(505),	    mov64_reg(1, 10), add64_imm(1, -8),
		 ldxdw(0, 1, 0),    jeq_imm(0, 0, 1), mov64_imm(0, 1),
		 make_word(0x95) };
}

std::string unique_temp_dir(const std::string &label)
{
	const std::string dir =
		std::filesystem::temp_directory_path().string() +
		"/bpftime_sass_aot_" + label + "_" +
		std::to_string(static_cast<long long>(::getpid()));
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	require(!ec, "cannot create temporary directory: " + ec.message());
	return dir;
}

std::string read_file(const std::string &path)
{
	std::ifstream ifs(path, std::ios::binary);
	require(ifs.good(), "cannot read " + path);
	std::ostringstream oss;
	oss << ifs.rdbuf();
	return oss.str();
}

void test_real_bpf_object_and_verifier()
{
	std::vector<uint64_t> words;
	std::string section;
	const auto error = load_bpf_program_words(
		SASS_AOT_SPIKE_BPF_OBJECT, kSpikeSection, words, section);
	require(!error, error.value_or("failed to load BPF program"));
	require(section == kSpikeSection, "loaded the wrong BPF section");
	require(!words.empty(), "loaded an empty BPF program");

	const auto verified = bpftime::verifier::gpu::verify_gpu_program(
		words.data(), words.size(), kSpikeSection);
	require(!verified,
		verified.value_or(
			"strict GPU verifier rejected the real fixture"));
}

void test_accepted_program_to_sass()
{
	std::vector<uint64_t> words;
	std::string section;
	const auto error = load_bpf_program_words(
		SASS_AOT_SPIKE_BPF_OBJECT, kSpikeSection, words, section);
	require(!error, error.value_or("failed to load BPF program"));

	SassAotOptions opts;
	// The space also proves subprocess arguments are passed without shell
	// tokenization.
	opts.out_dir = unique_temp_dir("accepted path");
	opts.func_name = kFuncName;
	const auto result = compile_ebpf_to_sass_aot(words, section, opts);
	require(result.ok, result.error);
	require(!result.verifier_rejected, "accepted program was rejected");
	require(std::filesystem::exists(result.ptx_path),
		"PTX was not written");
	require(std::filesystem::exists(result.cubin_path),
		"cubin was not written");

	const std::string ptx = read_file(result.ptx_path);
	require(ptx.find(".version " + opts.ptx_version) != std::string::npos,
		"PTX version header is missing");
	require(ptx.find(".target sm_120") != std::string::npos,
		"PTX sm_120 target is missing");
	require(ptx.find(".entry " + std::string(kFuncName)) !=
			std::string::npos,
		"compiled eBPF function was not promoted to an entry");

	const std::string sass = run_cuobjdump_sass(result.cubin_path, opts);
	require(sass.find("code for sm_120") != std::string::npos,
		"cuobjdump did not report sm_120 SASS:\n" + sass);
	require(sass.find("Function : " + std::string(kFuncName)) !=
			std::string::npos,
		"cuobjdump SASS did not contain the compiled function:\n" +
			sass);

	const std::string symbols =
		run_cuobjdump_symbols(result.cubin_path, opts);
	require(symbols.find(kFuncName) != std::string::npos,
		"cuobjdump symbols did not contain the compiled function:\n" +
			symbols);

	std::filesystem::remove_all(opts.out_dir);
}

void test_invalid_program_rejected_before_compilation()
{
	const std::string suffix =
		std::to_string(static_cast<long long>(::getpid()));
	const std::string marker =
		std::filesystem::temp_directory_path().string() +
		"/bpftime_sass_aot_ptxas_marker_" + suffix;
	const std::string fake_ptxas =
		std::filesystem::temp_directory_path().string() +
		"/bpftime_sass_aot_fake_ptxas_" + suffix;
	std::filesystem::remove(marker);
	{
		std::ofstream script(fake_ptxas);
		require(script.good(), "cannot create fake ptxas");
		script << "#!/bin/sh\ntouch '" << marker << "'\n";
	}
	require(::chmod(fake_ptxas.c_str(), 0755) == 0,
		"cannot make fake ptxas executable");

	SassAotOptions opts;
	opts.ptxas_path = fake_ptxas;
	opts.out_dir = unique_temp_dir("rejected");
	opts.func_name = "should_never_reach_ptxas";
	const auto result = compile_ebpf_to_sass_aot(
		make_lane_varying_branch_program(), "cuda__invalid", opts);
	require(!result.ok, "invalid program unexpectedly compiled");
	require(result.verifier_rejected,
		"invalid program was not rejected by the verifier");
	require(result.error.find("Warp-Uniform Branch Conditions") !=
			std::string::npos,
		"wrong verifier rejection: " + result.error);
	require(result.ptx_path.empty(),
		"invalid program reached PTX artifact creation");
	require(!std::filesystem::exists(std::filesystem::path(opts.out_dir) /
					 "sass_aot.ptx"),
		"invalid program wrote PTX");
	require(!std::filesystem::exists(std::filesystem::path(opts.out_dir) /
					 "sass_aot.cubin"),
		"invalid program wrote a cubin");
	require(!std::filesystem::exists(marker),
		"invalid program invoked ptxas");

	std::filesystem::remove(fake_ptxas);
	std::filesystem::remove(marker);
	std::filesystem::remove_all(opts.out_dir);
}

} // namespace

int main()
{
	try {
		test_real_bpf_object_and_verifier();
		test_accepted_program_to_sass();
		test_invalid_program_rejected_before_compilation();
		std::cout << "eBPF-to-SASS AOT spike tests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "eBPF-to-SASS AOT spike test failed: "
			  << error.what() << '\n';
		return 1;
	}
}
