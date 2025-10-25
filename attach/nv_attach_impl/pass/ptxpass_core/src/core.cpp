#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "json.hpp"

using nlohmann::json;

namespace ptxpass
{

static std::vector<std::regex>
compileRegexList(const std::vector<std::string> &patterns)
{
	std::vector<std::regex> out;
	out.reserve(patterns.size());
	for (const auto &p : patterns) {
		out.emplace_back(p, std::regex::ECMAScript);
	}
	return out;
}

static bool jsonGetBool(const json &j, const char *key, bool defVal)
{
	if (!j.is_object())
		return defVal;
	if (!j.contains(key))
		return defVal;
	try {
		return j.at(key).get<bool>();
	} catch (...) {
		return defVal;
	}
}

static std::string jsonGetString(const json &j, const char *key,
				 const std::string &defVal)
{
	if (!j.is_object())
		return defVal;
	if (!j.contains(key))
		return defVal;
	try {
		return j.at(key).get<std::string>();
	} catch (...) {
		return defVal;
	}
}

static int jsonGetInt(const json &j, const char *key, int defVal)
{
	if (!j.is_object())
		return defVal;
	if (!j.contains(key))
		return defVal;
	try {
		return j.at(key).get<int>();
	} catch (...) {
		return defVal;
	}
}

PassConfig JsonConfigLoader::loadFromFile(const std::string &path)
{
	std::ifstream ifs(path);
	if (!ifs.is_open()) {
		throw std::runtime_error("Failed to open config file: " + path);
	}
	json j;
	try {
		ifs >> j;
	} catch (const std::exception &e) {
		throw std::runtime_error(std::string("Failed to parse JSON: ") +
					 e.what());
	}

	PassConfig cfg;
	if (j.contains("name"))
		cfg.name = j["name"].get<std::string>();
	if (j.contains("description"))
		cfg.description = j["description"].get<std::string>();

	if (!j.contains("attach_points") || !j["attach_points"].is_object()) {
		throw std::runtime_error(
			"Invalid config: attach_points missing or not object");
	}
	auto &ap = j["attach_points"];
	if (ap.contains("includes"))
		cfg.attachPoints.includes =
			ap["includes"].get<std::vector<std::string>>();
	if (ap.contains("excludes"))
		cfg.attachPoints.excludes =
			ap["excludes"].get<std::vector<std::string>>();
	if (cfg.attachPoints.includes.empty()) {
		throw std::runtime_error(
			"Invalid config: attach_points.includes must have at least one regex");
	}
	if (j.contains("parameters"))
		cfg.parameters = j["parameters"];
	if (j.contains("validation"))
		cfg.validation = j["validation"];
	return cfg;
}

AttachPointMatcher::AttachPointMatcher(const AttachPoints &points)
	: includeRegexes(compileRegexList(points.includes)),
	  excludeRegexes(compileRegexList(points.excludes))
{
}

bool AttachPointMatcher::matches(const std::string &attachPoint) const
{
	bool included = false;
	for (const auto &r : includeRegexes) {
		if (std::regex_search(attachPoint, r)) {
			included = true;
			break;
		}
	}
	if (!included)
		return false;
	for (const auto &r : excludeRegexes) {
		if (std::regex_search(attachPoint, r)) {
			return false;
		}
	}
	return true;
}

std::string readAllFromStdin()
{
	std::ostringstream oss;
	oss << std::cin.rdbuf();
	if (!std::cin.good() && !std::cin.eof()) {
		throw std::runtime_error("Failed to read from stdin");
	}
	return oss.str();
}

bool isWhitespaceOnly(const std::string &s)
{
	for (char c : s) {
		if (!(c == ' ' || c == '\n' || c == '\r' || c == '\t' ||
		      c == '\v' || c == '\f')) {
			return false;
		}
	}
	return true;
}

std::string getEnv(const char *key)
{
	const char *v = std::getenv(key);
	return v ? std::string(v) : std::string();
}

std::pair<RuntimeInput, bool> parseRuntimeInput(const std::string &stdinData)
{
	RuntimeInput ri;
	try {
		auto j = json::parse(stdinData);
		if (j.contains("full_ptx"))
			ri.full_ptx = j["full_ptx"].get<std::string>();
		if (j.contains("to_patch_kernel"))
			ri.to_patch_kernel =
				j["to_patch_kernel"].get<std::string>();
		ri.global_ebpf_map_info_symbol = jsonGetString(
			j, "global_ebpf_map_info_symbol", "map_info");
		ri.ebpf_communication_data_symbol = jsonGetString(
			j, "ebpf_communication_data_symbol", "constData");
		return { ri, true };
	} catch (...) {
		ri.full_ptx = stdinData;
		return { ri, false };
	}
}

void emitRuntimeOutput(const std::string &outputPtx)
{
	json j;
	j["output_ptx"] = outputPtx;
	std::cout << j.dump();
}

// Validation: basic checks as placeholders
bool validateInput(const std::string &input, const json &validation)
{
	if (validation.is_null())
		return true;
	if (validation.contains("require_entry") &&
	    validation["require_entry"].get<bool>()) {
		if (!containsEntryFunction(input))
			return false;
	}
	if (validation.contains("require_ret") &&
	    validation["require_ret"].get<bool>()) {
		if (!containsRetInstruction(input))
			return false;
	}
	if (validation.contains("ptx_version_min")) {
		if (!validatePtxVersion(
			    input,
			    validation["ptx_version_min"].get<std::string>()))
			return false;
	}
	return true;
}

bool containsEntryFunction(const std::string &input)
{
	return input.find(".visible .entry") != std::string::npos;
}

bool containsRetInstruction(const std::string &input)
{
	return input.find("\n    ret;") != std::string::npos ||
	       input.find("\n\tret;") != std::string::npos;
}

bool validatePtxVersion(const std::string &input, const std::string &minVersion)
{
	// Expect line like: .version 7.0
	std::istringstream iss(input);
	std::string line;
	double want = 0.0;
	try {
		want = std::stod(minVersion);
	} catch (...) {
		return true;
	}
	while (std::getline(iss, line)) {
		if (line.rfind(".version", 0) == 0) {
			// parse number
			std::string v = line.substr(8);
			try {
				double have = std::stod(v);
				return have >= want;
			} catch (...) {
				return true;
			}
		}
	}
	return true;
}

bool hasMarker(const std::string &input, const std::string &marker)
{
	return input.find(marker) != std::string::npos;
}

std::pair<std::string, bool> instrumentEntry(const std::string &input,
					     const json &params)
{
	const std::string marker = "// __ptxpass_entry_injected__";
	if (hasMarker(input, marker))
		return { "", false };
	// Insert reg decl + timer read right after opening brace of the first
	// entry
	size_t entryPos = input.find(".visible .entry");
	if (entryPos == std::string::npos)
		return { "", false };
	size_t bracePos = input.find('{', entryPos);
	if (bracePos == std::string::npos)
		return { "", false };
	// Find end of line after '{'
	size_t insertPos = bracePos + 1;
	if (insertPos < input.size() && input[insertPos] == '\n') {
		insertPos += 1;
	} else {
		// Ensure new line after brace
		std::string out = input;
		out.insert(insertPos, "\n");
		// adjust positions
		insertPos += 1;
		// continue with out
		std::string regblk = marker +
				     "\n"
				     ".reg .u64 %ptxpass_e0;\n"
				     "mov.u64 %ptxpass_e0, %globaltimer;\n";
		out.insert(insertPos, regblk);
		return { out, true };
	}
	std::string out = input;
	const std::string saveStrategy =
		jsonGetString(params, "save_strategy", "minimal");
	const bool emitNops =
		jsonGetBool(params, "emit_nops_for_alignment", false);
	int padNops = jsonGetInt(params, "pad_nops", emitNops ? 1 : 0);
	std::ostringstream block;
	block << marker << "\n";
	// Save/restore占位：使用独立寄存器，避免影响原有寄存器
	block << ".reg .u64 %ptxpass_e0;\n";
	if (saveStrategy == "full") {
		block << ".reg .u64 %ptxpass_e1;\n"; // 可扩展更多保存
	}
	block << "mov.u64 %ptxpass_e0, %globaltimer;\n";
	while (padNops-- > 0)
		block << "nop;\n";
	out.insert(insertPos, block.str());
	return { out, true };
}

std::pair<std::string, bool> instrumentRetprobe(const std::string &input,
						const json &params)
{
	const std::string marker = "// __ptxpass_ret_injected__";
	if (hasMarker(input, marker))
		return { "", false };
	auto retPos = input.find("ret;");
	if (retPos == std::string::npos)
		return { "", false };
	std::string out = input;
	// Ensure reg decl exists near the start of function body
	const std::string regMarker = "// __ptxpass_regdecl_ret__";
	if (!hasMarker(out, regMarker)) {
		size_t entryPos = out.find(".visible .entry");
		if (entryPos != std::string::npos) {
			size_t bracePos = out.find('{', entryPos);
			if (bracePos != std::string::npos) {
				size_t insertPos = bracePos + 1;
				if (insertPos < out.size() &&
				    out[insertPos] == '\n')
					insertPos += 1;
				std::ostringstream decl;
				decl << regMarker << "\n";
				decl << ".reg .u64 %ptxpass_r0;\n";
				if (jsonGetString(params, "save_strategy",
						  "minimal") == "full") {
					decl << ".reg .u64 %ptxpass_r1;\n";
				}
				out.insert(insertPos, decl.str());
			}
		}
	}
	// Insert timer read (以及可选的对齐nop) before 'ret;'
	std::ostringstream inj;
	inj << marker << "\n"
	    << "mov.u64 %ptxpass_r0, %globaltimer;\n";
	if (jsonGetBool(params, "emit_nops_for_alignment", false)) {
		int padNops = jsonGetInt(params, "pad_nops", 1);
		while (padNops-- > 0)
			inj << "nop;\n";
	}
	out.insert(retPos, inj.str());
	return { out, true };
}

std::pair<std::string, bool> instrumentMemcapture(const std::string &input,
						  const json &params)
{
	const std::string marker = "// __ptxpass_memcapture_injected__";
	if (hasMarker(input, marker))
		return { "", false };
	size_t entryPos = input.find(".visible .entry");
	if (entryPos == std::string::npos)
		return { "", false };
	size_t bracePos = input.find('{', entryPos);
	if (bracePos == std::string::npos)
		return { "", false };
	size_t insertPos = bracePos + 1;
	if (insertPos < input.size() && input[insertPos] == '\n')
		insertPos += 1;
	std::string out = input;
	const bool emitNops =
		jsonGetBool(params, "emit_nops_for_alignment", false);
	int padNops = jsonGetInt(params, "pad_nops", emitNops ? 1 : 0);
	std::ostringstream block;
	block << marker << "\n";
	block << ".reg .u64 %ptxpass_m0;\n";
	block << "mov.u64 %ptxpass_m0, %globaltimer;\n";
	// 预留memcapture参数注入位点
	int bufferBytes = jsonGetInt(params, "buffer_bytes", 4096);
	int maxSegments = jsonGetInt(params, "max_segments", 4);
	(void)bufferBytes;
	(void)maxSegments; // 先占位，不破坏参数接口
	while (padNops-- > 0)
		block << "nop;\n";
	out.insert(insertPos, block.str());
	return { out, true };
}

std::pair<std::string, bool> instrumentMemcaptureAdvanced(
    const std::string &input,
    int bufferBytes,
    int maxSegments,
    bool allowPartial,
    const std::string &srcRegOrSymbol)
{
    const std::string marker = "// __ptxpass_memcapture_injected__";
    if (hasMarker(input, marker)) return {"", false};
    size_t entryPos = input.find(".visible .entry");
    if (entryPos == std::string::npos) return {"", false};
    size_t bracePos = input.find('{', entryPos);
    if (bracePos == std::string::npos) return {"", false};
    size_t insertPos = bracePos + 1; if (insertPos < input.size() && input[insertPos] == '\n') insertPos += 1;

    std::ostringstream oss;
    oss << marker << "\n";
    int alignedBytes = ((bufferBytes + 15) / 16) * 16;
    if (alignedBytes <= 0) alignedBytes = 16;
    // declare locals and regs
    oss << ".local .align 16 .b8 __ptxpass_mem_buf[" << alignedBytes << "];\n";
    oss << ".reg .u64 %ptxpass_m0, %ptxpass_m1, %ptxpass_m2;\n";
    oss << ".reg .pred %ptxpass_p0;\n";
    // compute addresses
    oss << "cvta.local.u64 %ptxpass_m0, __ptxpass_mem_buf;\n";
    oss << "mov.u64 %ptxpass_m1, " << srcRegOrSymbol << ";\n";
    // copy loop
    oss << "mov.u64 %ptxpass_m2, 0;\n";
    oss << "__ptxpass_mem_loop_check:\n";
    oss << "setp.ge.u64 %ptxpass_p0, %ptxpass_m2, " << alignedBytes << ";\n";
    oss << "@%ptxpass_p0 bra __ptxpass_mem_loop_end;\n";
    if (!allowPartial) {
        oss << "ld.global.v4.u32 {%r10,%r11,%r12,%r13}, [%ptxpass_m1+%ptxpass_m2];\n";
        oss << "st.local.v4.u32 [%ptxpass_m0+%ptxpass_m2], {%r10,%r11,%r12,%r13};\n";
    } else {
        oss << "// partial allowed: best-effort 16B\n";
        oss << "ld.global.v4.u32 {%r10,%r11,%r12,%r13}, [%ptxpass_m1+%ptxpass_m2];\n";
        oss << "st.local.v4.u32 [%ptxpass_m0+%ptxpass_m2], {%r10,%r11,%r12,%r13};\n";
    }
    oss << "add.u64 %ptxpass_m2, %ptxpass_m2, 16;\n";
    oss << "bra __ptxpass_mem_loop_check;\n";
    oss << "__ptxpass_mem_loop_end:\n";

    std::string out = input;
    out.insert(insertPos, oss.str());
    return { out, true };
}

} // namespace ptxpass
