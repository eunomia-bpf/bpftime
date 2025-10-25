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

// Try to parse JSON input; fallback to treating entire input as PTX when
// parsing fails Returns pair<RuntimeInput, bool isJsonMode>
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
					     const nlohmann::json &params);
std::pair<std::string, bool> instrumentRetprobe(const std::string &input,
						const nlohmann::json &params);
std::pair<std::string, bool> instrumentMemcapture(const std::string &input,
						  const nlohmann::json &params);

// Advanced memcapture with boundary checks and loop copy into a local buffer
std::pair<std::string, bool> instrumentMemcaptureAdvanced(
    const std::string &input,
    int bufferBytes,
    int maxSegments,
    bool allowPartial,
    const std::string &srcRegOrSymbol);

} // namespace ptxpass
