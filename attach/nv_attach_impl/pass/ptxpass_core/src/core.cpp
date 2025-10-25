#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <set>
#include <llvmbpf.hpp>
#include <llvm_jit_context.hpp>

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

// typed conversions are defined inline in header; avoid duplicate definitions
// here

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
	try {
		cfg = j.get<PassConfig>();
	} catch (const std::exception &e) {
		throw std::runtime_error(
			std::string("Invalid pass config schema: ") + e.what());
	}
	// minimal validation
	if (cfg.attachPoints.includes.empty()) {
		throw std::runtime_error(
			"Invalid config: attach_points.includes must have at least one regex");
	}
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
		ri = j.get<RuntimeInput>();
		return { ri, true };
	} catch (const std::exception &e) {
		std::cerr << "[ptxpass] Error: stdin must be JSON. "
			  << "Parse failed: " << e.what() << "\n";
		return { ri, false };
	} catch (...) {
		std::cerr << "[ptxpass] Error: stdin must be JSON.\n";
		return { ri, false };
	}
}

void emitRuntimeOutput(const std::string &outputPtx)
{
	RuntimeOutput ro{ outputPtx };
	json j = ro;
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

} // namespace ptxpass

namespace ptxpass
{
std::string filter_out_version_headers_ptx(const std::string &input)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	std::set<std::string> seen;
	while (std::getline(iss, line)) {
		bool skip = false;
		for (const auto &p : FILTERED_OUT_PREFIXES) {
			if (line.rfind(p, 0) == 0) {
				if (seen.contains(p))
					skip = true;
				else
					seen.insert(p);
				break;
			}
		}
		if (!skip)
			oss << line << '\n';
	}
	return oss.str();
}

std::string compile_ebpf_to_ptx_from_words(const std::vector<uint64_t> &words,
					   const std::string &target_sm)
{
	std::vector<ebpf_inst> insts;
	insts.reserve(words.size());
	for (auto w : words) {
		ebpf_inst ins{};
		ins.opcode = (uint8_t)(w & 0xFF);
		ins.dst = (uint8_t)((w >> 8) & 0xF);
		ins.src = (uint8_t)((w >> 12) & 0xF);
		ins.offset = (int16_t)((w >> 16) & 0xFFFF);
		ins.imm = (int32_t)(w >> 32);
		insts.push_back(ins);
	}
	bpftime::llvmbpf_vm vm;
	vm.unload_code();
	vm.load_code(insts.data(), insts.size() * 8);
	auto ptx = vm.generate_ptx(target_sm.c_str());
	return filter_out_version_headers_ptx(ptx.value_or(""));
}

std::pair<size_t, size_t> find_kernel_body(const std::string &ptx,
					   const std::string &kernel)
{
	static std::regex kernel_entry(
		R"(\.visible\s+\.entry\s+(\w+)\s*\(([^)]*)\))");
	std::smatch m;
	std::string::const_iterator search_start(ptx.cbegin());
	while (std::regex_search(search_start, ptx.cend(), m, kernel_entry)) {
		if (m[1] == kernel) {
			size_t begin = (size_t)(m[0].first - ptx.cbegin());
			size_t end = begin;
			std::vector<char> st;
			do {
				while (end < ptx.size() && ptx[end] != '{' &&
				       ptx[end] != '}')
					end++;
				if (end >= ptx.size())
					break;
				if (ptx[end] == '{')
					st.push_back('{');
				else
					st.pop_back();
				end++;
			} while (!st.empty());
			return { begin, end };
		}
		search_start = m.suffix().first;
	}
	return { std::string::npos, std::string::npos };
}

void log_transform_stats(const char *pass_name, int matched, size_t bytes_in,
			 size_t bytes_out)
{
	std::cerr << "[ptxpass] " << pass_name << ": matched=" << matched
		  << ", in=" << bytes_in << ", out=" << bytes_out << "\n";
}
} // namespace ptxpass