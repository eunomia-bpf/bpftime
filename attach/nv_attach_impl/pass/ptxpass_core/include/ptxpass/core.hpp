// Minimal core utilities for PTX pass executables
#pragma once

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include "json.hpp"
#include <fstream>
namespace ptxpass
{

namespace attach_points
{
struct AttachPoints {
	std::vector<std::string> includes;
	std::vector<std::string> excludes;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AttachPoints, includes, excludes);

} // namespace attach_points
namespace pass_config
{
struct PassConfig {
	std::string name;
	std::string description;
	attach_points::AttachPoints attach_points;
	int attach_type;
	nlohmann::json parameters; // optional
	nlohmann::json validation; // optional
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(PassConfig, name, description,
						attach_points, attach_type,
						parameters, validation);
} // namespace pass_config

class AttachPointMatcher {
    public:
	explicit AttachPointMatcher(const attach_points::AttachPoints &points);

	bool matches(const std::string &attachPoint) const;

    private:
	std::vector<std::regex> includeRegexes;
	std::vector<std::regex> excludeRegexes;
};

// Read entire stdin into a string; throws std::runtime_error on error
std::string read_all_from_stdin();

// Return true if s is empty or contains only whitespace (space, tab, newlines)
bool is_whitespace_only(const std::string &s);

// Fetch environment variable or empty string if not present
std::string get_env(const char *key);

// Standardized exit codes
enum ExitCode {
	Success = 0,
	ConfigError = 64,
	InputError = 65,
	TransformFailed = 66,
	UnknownError = 70,
};

// Runtime I/O (JSON over stdin/stdout)
namespace runtime_input
{
struct RuntimeInput {
	std::string full_ptx;
	std::string to_patch_kernel;
	std::string global_ebpf_map_info_symbol = "map_info";
	std::string ebpf_communication_data_symbol = "constData";
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RuntimeInput, full_ptx,
						to_patch_kernel,
						global_ebpf_map_info_symbol,
						ebpf_communication_data_symbol);
} // namespace runtime_input

// JSON stdout payload
namespace runtime_response
{
struct RuntimeResponse {
	std::string output_ptx;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RuntimeResponse, output_ptx);
} // namespace runtime_response

struct EbpfInstructionPair {
	uint32_t upper_32bit;
	uint32_t lower_32bit;
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(EbpfInstructionPair, upper_32bit,
				       lower_32bit);
	uint64_t to_uint64() const
	{
		return ((uint64_t)upper_32bit << 32) | lower_32bit;
	}
	EbpfInstructionPair(uint64_t inst = 0)
	{
		upper_32bit = inst >> 32;
		lower_32bit = inst & 0xffffffff;
	}
};

namespace runtime_request
{
struct RuntimeRequest {
	runtime_input::RuntimeInput input;
	std::vector<EbpfInstructionPair> ebpf_instructions;
	std::vector<uint64_t> get_uint64_ebpf_instructions() const
	{
		std::vector<uint64_t> result;
		for (const auto &inst : ebpf_instructions)
			result.push_back(inst.to_uint64());
		return result;
	}
	void set_ebpf_instructions(const std::vector<uint64_t> &words)
	{
		ebpf_instructions.clear();
		for (auto item : words) {
			ebpf_instructions.emplace_back(item);
		}
	}
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RuntimeRequest, input, ebpf_instructions);
} // namespace runtime_request

// Validation helpers
bool validate_input(const std::string &input, const nlohmann::json &validation);
bool contains_entry_function(const std::string &input);
bool contains_ret_instruction(const std::string &input);
bool validate_ptx_version(const std::string &input,
			  const std::string &minVersion);

// Shared utilities for PTX passes (refactored from legacy code)
// Filter out duplicate/irrelevant PTX headers
// (version/target/address_size/comments)
std::string filter_out_version_headers_ptx(const std::string &input);

// Compile eBPF (64-bit words encoding) to PTX function text (optionally target
// SM)
std::string compile_ebpf_to_ptx_from_words(
	const std::vector<uint64_t> &words, const std::string &target_sm,
	const std::string &func_name,
	bool add_register_guard_and_filter_version_headers,
	bool with_arguments);
std::string filter_compiled_ptx_for_ebpf_program(std::string input);
// Find kernel body range [begin, end) for a given kernel name using .visible
// .entry and brace depth Returns pair(begin,end); if not found, returns
// {std::string::npos, std::string::npos}
std::pair<size_t, size_t> find_kernel_body(const std::string &ptx,
					   const std::string &kernel);

// Emit simple stats to stderr (pass name, matched count, in/out sizes)
void log_transform_stats(const char *pass_name, int matched, size_t bytes_in,
			 size_t bytes_out);

static inline void emit_runtime_response_and_print(const std::string &str)
{
	using namespace runtime_response;
	RuntimeResponse output;
	nlohmann::json output_json;
	output.output_ptx = str;
	to_json(output_json, output);
	std::cout << output_json.dump();
}

static inline pass_config::PassConfig
load_pass_config_from_file(const std::filesystem::path &path)
{
	pass_config::PassConfig cfg;
	std::ifstream ifs(path);
	auto input_json = nlohmann::json::parse(ifs);
	pass_config::from_json(input_json, cfg);
	return cfg;
}

static inline runtime_request::RuntimeRequest
pass_runtime_request_from_string(const std::string &str)
{
	runtime_request::RuntimeRequest runtime_request;
	auto input_json = nlohmann::json::parse(str);
	runtime_request::from_json(input_json, runtime_request);
	return runtime_request;
}
} // namespace ptxpass

namespace bpftime::attach
{
std::string add_register_guard_for_ebpf_ptx_func(const std::string &ptxCode);

}
