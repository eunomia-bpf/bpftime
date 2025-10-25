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
	(void)input;
	(void)marker;
	return false;
}

std::pair<std::string, bool> instrumentEntry(const std::string &input,
					     const EntryParams &)
{
	return { "", false };
}

std::pair<std::string, bool> instrumentRetprobe(const std::string &input,
						const RetprobeParams &params)
{
	auto retPos = input.find("ret;");
	if (retPos == std::string::npos)
		return { "", false };
	std::string out = input;
	// Ensure reg decl exists near the start of function body
	size_t entryPos = out.find(".visible .entry");
	if (entryPos != std::string::npos) {
		size_t bracePos = out.find('{', entryPos);
		if (bracePos != std::string::npos) {
			size_t insertPos = bracePos + 1;
			if (insertPos < out.size() && out[insertPos] == '\n')
				insertPos += 1;
			std::ostringstream decl;
			decl << ".reg .u64 %ptxpass_r0;\n";
			if (params.save_strategy == "full") {
				decl << ".reg .u64 %ptxpass_r1;\n";
			}
			out.insert(insertPos, decl.str());
		}
	}
	// Insert timer read
	std::ostringstream inj;
	inj << "mov.u64 %ptxpass_r0, %globaltimer;\n";
	if (params.emit_nops_for_alignment) {
		int padNops = params.pad_nops;
		while (padNops-- > 0)
			inj << "nop;\n";
	}
	out.insert(retPos, inj.str());
	return { out, true };
}

std::pair<std::string, bool>
instrumentMemcapture(const std::string &input, const MemcaptureParams &params)
{
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
	const bool emitNops = params.emit_nops_for_alignment;
	int padNops = params.pad_nops;
	std::ostringstream block;
	block << ".reg .u64 %ptxpass_m0;\n";
	block << "mov.u64 %ptxpass_m0, %globaltimer;\n";

	int bufferBytes = params.buffer_bytes;
	int maxSegments = params.max_segments;
	(void)bufferBytes;
	(void)maxSegments; 
	while (padNops-- > 0)
		block << "nop;\n";
	out.insert(insertPos, block.str());
	return { out, true };
}

std::pair<std::string, bool>
instrumentMemcaptureAdvanced(const std::string &input, int bufferBytes,
			     int maxSegments, bool allowPartial,
			     const std::string &srcRegOrSymbol)
{
	size_t entryPos = input.find(".visible .entry");
	if (entryPos == std::string::npos)
		return { "", false };
	size_t bracePos = input.find('{', entryPos);
	if (bracePos == std::string::npos)
		return { "", false };
	size_t insertPos = bracePos + 1;
	if (insertPos < input.size() && input[insertPos] == '\n')
		insertPos += 1;

	std::ostringstream oss;
	int alignedBytes = ((bufferBytes + 15) / 16) * 16;
	if (alignedBytes <= 0)
		alignedBytes = 16;
	// declare locals and regs
	oss << ".local .align 16 .b8 __ptxpass_mem_buf[" << alignedBytes
	    << "];\n";
	oss << ".reg .u64 %ptxpass_m0, %ptxpass_m1, %ptxpass_m2;\n";
	oss << ".reg .pred %ptxpass_p0;\n";
	// compute addresses
	oss << "cvta.local.u64 %ptxpass_m0, __ptxpass_mem_buf;\n";
	oss << "mov.u64 %ptxpass_m1, " << srcRegOrSymbol << ";\n";
	// copy loop
	oss << "mov.u64 %ptxpass_m2, 0;\n";
	oss << "__ptxpass_mem_loop_check:\n";
	oss << "setp.ge.u64 %ptxpass_p0, %ptxpass_m2, " << alignedBytes
	    << ";\n";
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