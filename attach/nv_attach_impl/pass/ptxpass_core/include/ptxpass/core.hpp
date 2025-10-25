// Minimal core utilities for PTX pass executables
#pragma once

#include <regex>
#include <string>
#include <vector>
#include "json.hpp"

namespace ptxpass
{

struct AttachPoints {
	std::vector<std::string> includes;
	std::vector<std::string> excludes;
};

struct PassConfig {
	std::string name;
	std::string description;
	AttachPoints attachPoints;
	nlohmann::json parameters; // optional
	nlohmann::json validation; // optional
};

inline void from_json(const nlohmann::json &j, AttachPoints &ap)
{
	if (j.contains("includes"))
		ap.includes = j.at("includes").get<std::vector<std::string>>();
	if (j.contains("excludes"))
		ap.excludes = j.at("excludes").get<std::vector<std::string>>();
}

inline void from_json(const nlohmann::json &j, PassConfig &cfg)
{
	if (j.contains("name"))
		cfg.name = j.at("name").get<std::string>();
	if (j.contains("description"))
		cfg.description = j.at("description").get<std::string>();
	if (j.contains("attach_points"))
		cfg.attachPoints = j.at("attach_points").get<AttachPoints>();
	if (j.contains("parameters"))
		cfg.parameters = j.at("parameters");
	if (j.contains("validation"))
		cfg.validation = j.at("validation");
}

class JsonConfigLoader {
    public:
	// Parse JSON config from file path; throws std::runtime_error on error
	static PassConfig loadFromFile(const std::string &path);
};

class AttachPointMatcher {
    public:
	explicit AttachPointMatcher(const AttachPoints &points);

	bool matches(const std::string &attachPoint) const;

    private:
	std::vector<std::regex> includeRegexes;
	std::vector<std::regex> excludeRegexes;
};

// Read entire stdin into a string; throws std::runtime_error on error
std::string readAllFromStdin();

// Return true if s is empty or contains only whitespace (space, tab, newlines)
bool isWhitespaceOnly(const std::string &s);

// Fetch environment variable or empty string if not present
std::string getEnv(const char *key);

// Standardized exit codes
enum ExitCode {
	Success = 0,
	ConfigError = 64,
	InputError = 65,
	TransformFailed = 66,
	UnknownError = 70,
};

// Runtime I/O (JSON over stdin/stdout)
struct RuntimeInput {
	std::string full_ptx;
	std::string to_patch_kernel;
	std::string global_ebpf_map_info_symbol;
	std::string ebpf_communication_data_symbol;
};

// Parameter structs for typed deserialization
struct EntryParams {
	std::string save_strategy = "minimal"; // "minimal" or "full"
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
};

struct RetprobeParams {
	std::string save_strategy = "minimal"; // "minimal" or "full"
	bool emit_nops_for_alignment = false;
	int pad_nops = 1;
};

struct MemcaptureParams {
	int buffer_bytes = 4096;
	int max_segments = 4;
	bool allow_partial = true;
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
};

// Try to parse JSON input using typed deserialization; fallback to plain PTX
// Returns pair<RuntimeInput, bool isJsonMode>
std::pair<RuntimeInput, bool> parseRuntimeInput(const std::string &stdinData);

// Emit JSON with {"output_ptx": "..."}
void emitRuntimeOutput(const std::string &outputPtx);

// Validation helpers
bool validateInput(const std::string &input, const nlohmann::json &validation);
bool containsEntryFunction(const std::string &input);
bool containsRetInstruction(const std::string &input);
bool validatePtxVersion(const std::string &input,
			const std::string &minVersion);

// Markers and transforms (return {output, modified})
bool hasMarker(const std::string &input, const std::string &marker);
std::pair<std::string, bool> instrumentEntry(const std::string &input,
					     const EntryParams &params);
std::pair<std::string, bool> instrumentRetprobe(const std::string &input,
						const RetprobeParams &params);
std::pair<std::string, bool>
instrumentMemcapture(const std::string &input, const MemcaptureParams &params);

// Advanced memcapture with boundary checks and loop copy into a local buffer
std::pair<std::string, bool>
instrumentMemcaptureAdvanced(const std::string &input, int bufferBytes,
			     int maxSegments, bool allowPartial,
			     const std::string &srcRegOrSymbol);

// nlohmann::json arbitrary type conversions for runtime and params
inline void from_json(const nlohmann::json &j, RuntimeInput &ri)
{
	j.at("full_ptx").get_to(ri.full_ptx);
	if (j.contains("to_patch_kernel"))
		j.at("to_patch_kernel").get_to(ri.to_patch_kernel);
	ri.global_ebpf_map_info_symbol =
		j.value("global_ebpf_map_info_symbol", std::string("map_info"));
	ri.ebpf_communication_data_symbol = j.value(
		"ebpf_communication_data_symbol", std::string("constData"));
}

// Shared utilities for PTX passes (refactored from legacy code)
// Filter out duplicate/irrelevant PTX headers
// (version/target/address_size/comments)
std::string filter_out_version_headers_ptx(const std::string &input);

// Compile eBPF (64-bit words encoding) to PTX function text (optionally target
// SM)
std::string compile_ebpf_to_ptx_from_words(const std::vector<uint64_t> &words,
					   const std::string &target_sm);

// Find kernel body range [begin, end) for a given kernel name using .visible
// .entry and brace depth Returns pair(begin,end); if not found, returns
// {std::string::npos, std::string::npos}
std::pair<size_t, size_t> find_kernel_body(const std::string &ptx,
					   const std::string &kernel);

// Emit simple stats to stderr (pass name, matched count, in/out sizes)
void log_transform_stats(const char *pass_name, int matched, size_t bytes_in,
			 size_t bytes_out);

inline void from_json(const nlohmann::json &j, EntryParams &p)
{
	p.save_strategy = j.value("save_strategy", std::string("minimal"));
	p.emit_nops_for_alignment = j.value("emit_nops_for_alignment", false);
	p.pad_nops = j.value("pad_nops", 0);
}

inline void from_json(const nlohmann::json &j, RetprobeParams &p)
{
	p.save_strategy = j.value("save_strategy", std::string("minimal"));
	p.emit_nops_for_alignment = j.value("emit_nops_for_alignment", false);
	p.pad_nops = j.value("pad_nops", 1);
}

inline void from_json(const nlohmann::json &j, MemcaptureParams &p)
{
	p.buffer_bytes = j.value("buffer_bytes", 4096);
	p.max_segments = j.value("max_segments", 4);
	p.allow_partial = j.value("allow_partial", true);
	p.emit_nops_for_alignment = j.value("emit_nops_for_alignment", false);
	p.pad_nops = j.value("pad_nops", 0);
}

} // namespace ptxpass
