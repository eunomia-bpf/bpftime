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

std::string filter_out_version_headers(const std::string &input)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	std::set<std::string> found_patterns;

	while (std::getline(iss, line)) {
		bool skip = false;
		for (const auto &s : FILTERED_OUT_PREFIXES) {
			if (line.starts_with(s)) {
				if (found_patterns.contains(s))
					skip = true;
				else {
					found_patterns.insert(s);
				}
				break;
			}
		}
		if (!skip)
			oss << line << std::endl;
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

static void test_func()
{
}

static std::string generate_ptx_for_ebpf(const std::vector<ebpf_inst> &inst,
					 const std::string &func_name,
					 bool with_arguments)
{
	llvmbpf_vm vm;
	vm.register_external_function(1, "map_lookup", (void *)test_func);
	vm.register_external_function(2, "map_update", (void *)test_func);
	vm.register_external_function(3, "map_delete", (void *)test_func);
	vm.register_external_function(6, "print", (void *)test_func);
	vm.register_external_function(14, "get_pid_tgid", (void *)test_func);
	vm.register_external_function(25, "perf_event_output", (void *)test_func);

	vm.register_external_function(501, "puts", (void *)test_func);
	vm.register_external_function(502, "get_global_timer",
				      (void *)test_func);
	vm.register_external_function(503, "get_block_idx", (void *)test_func);
	vm.register_external_function(504, "get_block_dim", (void *)test_func);
	vm.register_external_function(505, "get_thread_idx", (void *)test_func);

	vm.load_code(inst.data(), inst.size() * 8);
	llvm_bpf_jit_context ctx(vm);
	SPDLOG_INFO(
		"Compiling eBPF to PTX {}, eBPF instructions count {}, with arguments {}",
		func_name, inst.size(), with_arguments);
	auto original_ptx = *ctx.generate_ptx(with_arguments);
	if (spdlog::get_level() <= SPDLOG_LEVEL_DEBUG) {
		auto path = "/tmp/dump-ebpf." + func_name + ".ptx";

		std::ofstream ofs(path);
		ofs << original_ptx << std::endl;
		SPDLOG_DEBUG("Dumped {}", path);
	}
	auto filtered_ptx = add_register_guard_for_ebpf_ptx_func(
		filter_compiled_ptx_for_ebpf_program(original_ptx, func_name));

	return filtered_ptx;
}
std::optional<std::string>
nv_attach_impl::patch_with_memcapture(std::string input,
				      const nv_attach_entry &entry,
				      bool should_set_trampoline)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version",
		".target",
		".address_size",
	};
	SPDLOG_INFO("Patching memcapture: input size {}", input.size());
	static std::regex pattern(
		R"(^\s*(ld|st)\.(const|global|local|param)?\.(((s|u|b)(8|16|32|64))|\.b128|(\.f(16|16x2|32|64))) +(.+), *(.+);\s*$)");
	std::ostringstream function_def;

	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	int count = 0;
	while (std::getline(iss, line)) {
		bool skip = false;
		for (const auto &prefix : FILTERED_OUT_PREFIXES) {
			if (line.starts_with(prefix)) {
				skip = true;
				break;
			}
		}
		if (!skip)
			oss << line << std::endl;
		if (std::regex_match(line, pattern)) {
			SPDLOG_DEBUG("Found matched line {}", line);
			count++;

			auto insts = entry.instuctions;

			{
				SPDLOG_INFO(
					"Generating trampoline for ebpf entry..");
				std::vector<ebpf_inst> entry;
				int32_t total_length;
				{
					auto N = line.size() + 1;
					total_length = N / 8 * 8 +
						       ((N % 8 != 0) ? 8 : 0);
				}
				assert(total_length % 8 == 0);
				// *(u64) (r10 - 8) = 0
				entry.push_back(
					ebpf_inst{ .opcode = EBPF_OP_STDW,
						   .dst = 10,
						   .src = 0,
						   .offset = -8,
						   .imm = 0 });
				// r10 -= N
				entry.push_back(
					ebpf_inst{ .opcode = EBPF_OP_SUB64_IMM,
						   .dst = 10,
						   .src = 0,
						   .offset = 0,
						   .imm = total_length });
				// r1 = r10
				entry.push_back(
					ebpf_inst{ .opcode = EBPF_OP_MOV64_REG,
						   .dst = 1,
						   .src = 10,
						   .offset = 0,
						   .imm = 0 });

				for (int i = 0; i < line.size(); i += 8) {
					uint64_t curr_word = 0;
					for (int j = 0;
					     j < 8 && (j + i) < line.size();
					     j++) {
						curr_word |=
							(((uint64_t)line[i + j])
							 << (j * 8));
					}
					// r2 = <uint64_t>curr_word
					entry.push_back(ebpf_inst{
						.opcode = EBPF_OP_LDDW,
						.dst = 2,
						.src = 0,
						.offset = 0,
						.imm = (int32_t)(uint32_t)
							curr_word });
					entry.push_back(ebpf_inst{
						.opcode = 0,
						.dst = 0,
						.src = 0,
						.offset = 0,
						.imm = (int32_t)(uint32_t)(curr_word >>
									   32) });

					// *(u64) (r1 + i) = r2
					entry.push_back(ebpf_inst{
						.opcode = EBPF_OP_STXDW,
						.dst = 1,
						.src = 2,
						.offset = (int16_t)(i),
						.imm = 0 });
				}
				insts.insert(insts.begin(), entry.begin(),
					     entry.end());
			}

			auto probe_func_name = memcapture_func_name(count);
			auto filtered_ptx = generate_ptx_for_ebpf(
				insts, probe_func_name, true);
			function_def << filtered_ptx << std::endl;
			oss << "call " << probe_func_name << ";" << std::endl;
		}
	}
	auto result = function_def.str() + "\n" + oss.str();
	// if (should_set_trampoline)
	// 	result = wrap_ptx_with_trampoline(result);
	SPDLOG_INFO("Patched {} instructions. output size {}", count,
		    result.size());
	return result;
}

std::optional<std::string>
nv_attach_impl::patch_with_probe_and_retprobe(std::string ptx,
					      const nv_attach_entry &entry,
					      bool should_set_trampoline)
{
	const auto &probe_detail =
		std::get<nv_attach_function_probe>(entry.type);
	std::vector<std::string> targets = entry.kernels;
	if (targets.empty())
		targets.push_back(probe_detail.func);

	for (const auto &target : targets) {
		static std::regex kernel_entry_finder(
			R"(\.visible\s+\.entry\s+(\w+)\s*\(([^)]*)\))");
		struct kernel_section {
			std::string name;
			size_t begin;
			size_t end;
		};
		std::vector<kernel_section> kernels;

		std::smatch match;
		std::string::const_iterator search_start(ptx.cbegin());
		while (std::regex_search(search_start, ptx.cend(), match,
					 kernel_entry_finder)) {
			kernels.push_back(kernel_section{
				.name = match[1],
				.begin =
					(size_t)(match[0].first - ptx.cbegin()),
				.end = 0 });
			search_start = match.suffix().first;
		}
		for (auto &kernel_sec : kernels) {
			std::vector<char> stack;
			size_t idx = kernel_sec.begin;
			do {
				while (ptx[idx] != '{' && ptx[idx] != '}')
					idx++;
				if (ptx[idx] == '{')
					stack.push_back('{');
				else {
					stack.pop_back();
				}
				idx++;
			} while (!stack.empty());
			kernel_sec.end = idx;
			if (kernel_sec.name == target) {
				auto probe_func_name =
					(probe_detail.is_retprobe ?
						 std::string(
							 "__retprobe_func__") :
						 std::string(
							 "__probe_func__")) +
					target;
				auto compiled_ebpf_ptx =
					generate_ptx_for_ebpf(entry.instuctions,
							      probe_func_name,
							      false);
				std::string sub_str = ptx.substr(
					kernel_sec.begin,
					kernel_sec.end - kernel_sec.begin);
				if (probe_detail.is_retprobe) {
					static std::regex ret_pattern(
						R"((\s+)(ret;))");
					sub_str = std::regex_replace(
						sub_str, ret_pattern,
						"$1call " + probe_func_name +
							";\n$1$2");
				} else {
					static std::regex begin_pattern(
						R"((\{)(\s*\.reg|\s*\.shared|\s*$))");
					sub_str = std::regex_replace(
						sub_str, begin_pattern,
						"$1\n    call " +
							probe_func_name +
							";\n$2");
				}
				ptx = ptx.replace(kernel_sec.begin,
						  kernel_sec.end -
							  kernel_sec.begin,
						  sub_str);
				ptx = compiled_ebpf_ptx + "\n" + ptx;
				break;
			}
		}
	}

	ptx = filter_out_version_headers(ptx);
	return ptx;
}
