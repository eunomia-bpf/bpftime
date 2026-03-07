#include "ptxpass/core.hpp"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <set>
#include <string_view>
#include <llvmbpf.hpp>
#include <llvm_jit_context.hpp>

#include "json.hpp"

using nlohmann::json;

namespace ptxpass
{

template <typename Fn>
static void for_each_line(std::string_view input, Fn &&fn)
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

static std::vector<std::regex>
compile_regex_list(const std::vector<std::string> &patterns)
{
	std::vector<std::regex> out;
	out.reserve(patterns.size());
	for (const auto &p : patterns) {
		out.emplace_back(p, std::regex::ECMAScript);
	}
	return out;
}

AttachPointMatcher::AttachPointMatcher(const attach_points::AttachPoints &points)
	: includeRegexes(compile_regex_list(points.includes)),
	  excludeRegexes(compile_regex_list(points.excludes))
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

std::string read_all_from_stdin()
{
	std::string input(std::istreambuf_iterator<char>(std::cin), {});
	if (!std::cin.good() && !std::cin.eof()) {
		throw std::runtime_error("Failed to read from stdin");
	}
	return input;
}

bool is_whitespace_only(const std::string &s)
{
	for (char c : s) {
		if (!(c == ' ' || c == '\n' || c == '\r' || c == '\t' ||
		      c == '\v' || c == '\f')) {
			return false;
		}
	}
	return true;
}

std::string get_env(const char *key)
{
	const char *v = std::getenv(key);
	return v ? std::string(v) : std::string();
}

// Validation: basic checks as placeholders
bool validate_input(const std::string &input, const json &validation)
{
	if (validation.is_null())
		return true;
	if (validation.contains("require_entry") &&
	    validation["require_entry"].get<bool>()) {
		if (!contains_entry_function(input))
			return false;
	}
	if (validation.contains("require_ret") &&
	    validation["require_ret"].get<bool>()) {
		if (!contains_ret_instruction(input))
			return false;
	}
	if (validation.contains("ptx_version_min")) {
		if (!validate_ptx_version(
			    input,
			    validation["ptx_version_min"].get<std::string>()))
			return false;
	}
	return true;
}

bool contains_entry_function(const std::string &input)
{
	return input.find(".visible .entry") != std::string::npos;
}

bool contains_ret_instruction(const std::string &input)
{
	return input.find("\n    ret;") != std::string::npos ||
	       input.find("\n\tret;") != std::string::npos;
}

bool validate_ptx_version(const std::string &input,
			  const std::string &minVersion)
{
	// Expect line like: .version 7.0
	double want = 0.0;
	try {
		want = std::stod(minVersion);
	} catch (...) {
		return true;
	}
	std::optional<bool> version_ok;
	for_each_line(input, [&](std::string_view line) {
		if (version_ok.has_value())
			return;
		if (line.rfind(".version", 0) == 0) {
			// parse number
			std::string v(line.substr(8));
			try {
				double have = std::stod(v);
				version_ok = have >= want;
			} catch (...) {
				version_ok = true;
			}
		}
	});
	return version_ok.value_or(true);
}

} // namespace ptxpass

namespace ptxpass
{
std::string filter_out_version_headers_ptx(const std::string &input)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};

	std::set<std::string> seen;
	std::string filtered;
	filtered.reserve(input.size());
	for_each_line(input, [&](std::string_view line) {
		bool skip = false;
		for (const auto &p : FILTERED_OUT_PREFIXES) {
			if (line.rfind(p, 0) == 0) {
				if (seen.contains(p)) {
					skip = true;
				} else {
					seen.insert(p);
				}
				break;
			}
		}
		if (!skip)
			filtered.append(line).push_back('\n');
	});
	return filtered;
}
static uint64_t test_func(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	return 0;
}

std::string compile_ebpf_to_ptx_from_words(
	const std::vector<uint64_t> &words, const std::string &target_sm,
	const std::string &func_name,
	bool add_register_guard_and_filter_version_headers, bool with_arguments)
{
	const ebpf_inst *insts =
		reinterpret_cast<const ebpf_inst *>(words.data());
	size_t insts_count = words.size();
	bpftime::llvmbpf_vm vm;
	vm.register_external_function(1, "map_lookup", (void *)test_func);
	vm.register_external_function(2, "map_update", (void *)test_func);
	vm.register_external_function(3, "map_delete", (void *)test_func);
	vm.register_external_function(6, "print", (void *)test_func);
	vm.register_external_function(14, "get_pid_tgid", (void *)test_func);
	vm.register_external_function(25, "perf_event_output",
				      (void *)test_func);

	vm.register_external_function(501, "puts", (void *)test_func);
	vm.register_external_function(502, "get_global_timer",
				      (void *)test_func);
	vm.register_external_function(503, "get_block_idx", (void *)test_func);
	vm.register_external_function(504, "get_block_dim", (void *)test_func);
	vm.register_external_function(505, "get_thread_idx", (void *)test_func);
	vm.register_external_function(507, "cuda_exit", (void *)test_func);
	vm.register_external_function(508, "get_grid_dim", (void *)test_func);
	vm.register_external_function(509, "get_sm_id", (void *)test_func);
	vm.register_external_function(510, "get_warp_id", (void *)test_func);
	vm.register_external_function(511, "get_lane_id", (void *)test_func);

	vm.load_code(insts, insts_count * sizeof(ebpf_inst));
	bpftime::llvm_bpf_jit_context ctx(vm);
	std::string original_ptx;
	if (auto optional_ptx = ctx.generate_ptx(with_arguments, func_name,
						 target_sm.c_str());
	    optional_ptx) {
		original_ptx = *optional_ptx;
	} else {
		SPDLOG_ERROR("Unable to produce PTX from eBPF");
		throw std::runtime_error("Unable to produce PTX from eBPF");
	}
	std::string filtered_ptx;
	if (add_register_guard_and_filter_version_headers) {
		filtered_ptx =
			bpftime::attach::add_register_guard_for_ebpf_ptx_func(
				filter_compiled_ptx_for_ebpf_program(
					original_ptx));
	} else {
		filtered_ptx = original_ptx;
	}
	return filtered_ptx;
}
std::string filter_compiled_ptx_for_ebpf_program(std::string input)
{
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
	std::string filtered;
	filtered.reserve(input.size());
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

	return result;
}
std::pair<size_t, size_t> find_kernel_body(const std::string &ptx,
					   const std::string &kernel)
{
	static std::regex kernel_entry(
		R"((\.visible\s+)?\.entry\s+(\w+)\s*\(([^)]*)\))");
	std::smatch m;
	std::string::const_iterator search_start(ptx.cbegin());
	while (std::regex_search(search_start, ptx.cend(), m, kernel_entry)) {
		if (m[2] == kernel) {
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
	SPDLOG_INFO("[ptxpass] {}: matched={}, in={}, out={}", pass_name,
		    matched, bytes_in, bytes_out);
}
} // namespace ptxpass
