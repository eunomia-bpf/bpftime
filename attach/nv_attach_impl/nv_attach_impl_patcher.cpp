#include "ebpf_inst.h"
#include "llvm_jit_context.hpp"
#include "llvmbpf.hpp"
#include "nv_attach_impl.hpp"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
using namespace bpftime;
using namespace attach;

static std::string memcapture_func_name(int idx)
{
	return std::string("__memcapture__") + std::to_string(idx);
}

namespace bpftime::attach
{

std::string add_semicolon_for_variable_lines(std::string input)
{
	std::istringstream iss(input);
	std::string line;
	std::ostringstream oss;
	while (std::getline(iss, line)) {
		while (!line.empty() && line.ends_with("\n"))
			line.pop_back();
		oss << line;
		if ((line.starts_with(".align") ||
		     line.starts_with(".global")) &&
		    !line.ends_with(";") && line.ends_with("}")) {
			oss << ";";
			SPDLOG_DEBUG("Patching line: {}", line);
		}
		oss << std::endl;
	}
	return oss.str();
}

std::string filter_compiled_ptx_for_ebpf_program(std::string input,
						 std::string new_func_name)
{
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};
	static const std::regex FILTERED_OUT_REGEXS[] = {
		std::regex(
			R"(\.extern\s+\.func\s+\(\s*\.param\s+\.b64\s+func_retval0\s*\)\s+_bpf_helper_ext_\d{4}\s*\(\s*(?:\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*,\s*)*\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*\)\s*;)"),

	};
	static const std::string FILTERED_OUT_SECTION[] = {
		R"(.visible .func bpf_main(
	.param .b64 bpf_main_param_0,
	.param .b64 bpf_main_param_1
))",
		R"(.visible .func bpf_main())"
	};
	while (std::getline(iss, line)) {
		// if(line.starts_with)
		bool skip = false;
		for (const auto &prefix : FILTERED_OUT_PREFIXES) {
			if (line.starts_with(prefix)) {
				skip = true;
				break;
			}
		}
		if (!skip)
			oss << line << std::endl;
	}
	auto result = oss.str();
	for (const auto &sec : FILTERED_OUT_SECTION) {
		if (auto pos = result.find(sec); pos != result.npos) {
			result = result.replace(pos, sec.size(), "");
		}
	}
	for (const auto &regex : FILTERED_OUT_REGEXS) {
		result = std::regex_replace(result, regex, "");
	}

	return ".func " + new_func_name + " " + result;
}

} // namespace bpftime::attach
