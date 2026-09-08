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
	// Opt-in automatic warp-uniform execution. When true, the pass lowers
	// hook-site handler invocations to verifier-admitted per-warp leader
	// execution instead of one scalar call per active thread. The runtime
	// only sets this after the program passed full verification and the
	// warp-policy eligibility analysis, so the flag is admission-backed
	// rather than caller-provided trust.
	bool warp_auto_execution = false;
	// Opt-in diagnostic: the pass predicated-adds module global
	// kWarpHookCallCountGlobal at each generated warp-leader call site so
	// the runtime can read back the actual invocation count. The runtime
	// sets it only together with warp_auto_execution; it never changes the
	// BPF object or the warp-policy eligibility decision.
	bool warp_hook_call_count = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
	RuntimeInput, full_ptx, to_patch_kernel, global_ebpf_map_info_symbol,
	ebpf_communication_data_symbol, warp_auto_execution,
	warp_hook_call_count);
} // namespace runtime_input

// JSON stdout payload
namespace runtime_response
{
struct RuntimeResponse {
	std::string output_ptx;
	bool modified;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RuntimeResponse, output_ptx, modified);
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

// ------- Automatic warp-execution lowering helpers -------
//
// When RuntimeInput::warp_auto_execution is set, the hook-site handler
// invocation is lowered to warp-leader execution: the appended preamble
// elects the lowest-numbered lane that actually executes this hook site
// (partial warps and predicated exits stay correct because every lane of the
// site contributes its own hook predicate to a warp ballot), and only that
// elected leader performs the handler call. This lowers the verifier-admitted
// uniform policy contract; programs with per-lane observable side effects are
// not admitted (see the runtime-side eligibility analysis).

// True when the PTX module advertises a PTX ISA version that supports
// vote.ballot.sync (>= 6.0).
bool ptx_supports_warp_sync(const std::string &ptx);

// Register declaration lines that must be inserted once at the top of the
// patched kernel body so every hook site can use fresh bpftime-only symbol
// names. Each emitted line ends with a newline.
std::string warp_execution_register_decls();

// Insert warp_execution_register_decls() at the start of the kernel body.
// Returns false (without modifying ptx) if the kernel could not be located.
bool insert_warp_execution_register_decls(std::string &ptx,
					      const std::string &kernel);

// Opt-in actual-invocation counter for the automatic warp-execution
// lowering. When the runtime enables it (environment
// kWarpHookCallCountEnv, non-empty value other than "0") and the warp
// leader lowering applies, each elected leader predicated-adds 1 to the
// module global below at the generated call site. The runtime resolves the
// global in the loaded module and reads the value back to the host.
inline constexpr const char *kWarpHookCallCountGlobal =
	"bpftime_warp_hook_call_count";
inline constexpr const char *kWarpHookCallCountEnv =
	"BPFTIME_GPU_WARP_HOOK_CALL_COUNT";

// Environment predicate matching the auto-warp enablement convention:
// non-empty value other than "0".
inline bool warp_hook_call_count_env_enabled()
{
	const std::string value = get_env(kWarpHookCallCountEnv);
	return !value.empty() && value != "0";
}

// Module-scope declaration of the counter global, emitted only when at
// least one warp-leader call site carries the counter increment.
std::string warp_hook_call_count_global_decl();

// Election preamble plus a single leader-predicated `call func_name;` for one
// ret/exit site. pred_text is the original site predicate with trailing
// space, e.g. "@%p1 " (may be empty for unconditional sites). The original
// control-flow line must still be emitted after the returned text.
// When count_hook_calls is true, a predicated red-add on
// kWarpHookCallCountGlobal is emitted under the same leader predicate,
// immediately before the call.
std::string emit_warp_leader_hook_prefix(const std::string &func_name,
					 const std::string &pred_text,
					 bool count_hook_calls = false);

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

static inline std::string
emit_runtime_response_and_return(const std::string &str, bool modified)
{
	using namespace runtime_response;
	RuntimeResponse output;
	nlohmann::json output_json;
	output.output_ptx = str;
	output.modified = modified;
	to_json(output_json, output);
	return output_json.dump();
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
