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
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <fstream>
using namespace bpftime;
using namespace attach;

namespace bpftime::attach
{

namespace {
template <typename Fn>
void for_each_line(std::string_view input, Fn &&fn)
{
	size_t start = 0;
	while (start < input.size()) {
		const auto end = input.find('\n', start);
		if (end == std::string_view::npos) {
			fn(input.substr(start));
			break;
		}
		fn(input.substr(start, end - start));
		start = end + 1;
	}
}
} // namespace

std::string add_semicolon_for_variable_lines(std::string input)
{
	std::string result;
	result.reserve(input.size() + 8);
	for_each_line(input, [&](std::string_view line) {
		result.append(line);
		if ((line.starts_with(".align") ||
		     line.starts_with(".global")) &&
		    !line.ends_with(";") && line.ends_with("}")) {
			result.push_back(';');
			SPDLOG_DEBUG("Patching line: {}", line);
		}
		result.push_back('\n');
	});
	return result;
}

std::string filter_compiled_ptx_for_ebpf_program(std::string input,
						 std::string new_func_name)
{
	std::string filtered;
	filtered.reserve(input.size());
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
	for_each_line(input, [&](std::string_view line) {
		// if(line.starts_with)
		bool skip = false;
		for (const auto &prefix : FILTERED_OUT_PREFIXES) {
			if (line.starts_with(prefix)) {
				skip = true;
				break;
			}
		}
		if (!skip)
			filtered.append(line).push_back('\n');
	});
	auto result = std::move(filtered);
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
