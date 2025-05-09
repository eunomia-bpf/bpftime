#include "ebpf_inst.h"
#include "llvm_jit_context.hpp"
#include "llvmbpf.hpp"
#include "nv_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <fstream>
#include <ostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
using namespace bpftime;
using namespace attach;

namespace bpftime::attach
{
std::string filter_compiled_ptx_for_ebpf_program(std::string input)
{
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size"
	};
	static const std::regex FILTERED_OUT_REGEXS[] = {
		std::regex(
			R"(\.extern\s+\.func\s+\(\s*\.param\s+\.b64\s+func_retval0\s*\)\s+_bpf_helper_ext_\d{4}\s*\(\s*(?:\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*,\s*)*\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*\)\s*;)")
	};
	static const std::string FILTERED_OUT_SECTION[] = {
		R"(.visible .func bpf_main(
	.param .b64 bpf_main_param_0,
	.param .b64 bpf_main_param_1
))"
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
	return result;
}
} // namespace bpftime::attach

static void test_func()
{
}
std::optional<std::string>
nv_attach_impl::patch_with_memcapture(std::string input,
				      const nv_attach_entry &entry)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version",
		".target",
		".address_size",
	};
	SPDLOG_INFO("Patching memcapture: input size {}", input.size());
	static std::regex pattern(
		R"(^\s*(ld|st)\.(const|global|local|param)?\.(((s|u|b)(8|16|32|64))|\.b128|(\.f(16|16x2|32|64))) +(.+), *(.+);\s*$)");
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
			llvmbpf_vm vm;
			vm.register_external_function(1, "map_lookup",
						      (void *)test_func);
			vm.register_external_function(2, "map_update",
						      (void *)test_func);
			vm.register_external_function(3, "map_delete",
						      (void *)test_func);
			vm.register_external_function(6, "print",
						      (void *)test_func);

			auto insts = entry.instuctions;

			{
				SPDLOG_INFO(
					"Generating trampoline for ebpf entry..");
				std::vector<ebpf_inst> entry;
				// Set N = ceil(line.size() + 1 )
				int32_t total_length =
					(line.size() + 1 + 3) / 4;
				// *(u32) (r10 - 4) = 0
				entry.push_back(
					ebpf_inst{ .opcode = EBPF_OP_STW,
						   .dst = 10,
						   .src = 0,
						   .offset = -4,
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

				for (int i = 0; i < line.size(); i += 4) {
					uint32_t curr_word = 0;
					for (int j = 0;
					     j < 4 && (j + i) < line.size();
					     j++) {
						curr_word |=
							(((uint32_t)line[i + j])
							 << (j * 8));
					}
					// *(u32) (r1 + i) = WORD
					entry.push_back(ebpf_inst{
						.opcode = EBPF_OP_STW,
						.dst = 1,
						.src = 0,
						.offset = (int16_t)(i),
						.imm = (int32_t)curr_word });
				}
				insts.insert(insts.begin(), entry.begin(),
					     entry.end());
			}

			vm.load_code(insts.data(), insts.size() * 8);
			llvm_bpf_jit_context ctx(vm);
			auto original_ptx = *ctx.generate_ptx();

			auto filtered_ptx =
				filter_compiled_ptx_for_ebpf_program(
					original_ptx);

			oss << filtered_ptx << std::endl;
		}
	}
    /**
    TODO: 每个probe存成一个单独的函数，当probe触发时，去调用这个函数  
    给filter完成的probe ptx加一个函数头，然后塞到最前面即可
    ld指令后面生成调用对应函数的代码即可   
     * 
     */
	// auto result = oss.str();
	auto result = wrap_ptx_with_trampoline(oss.str());
	SPDLOG_INFO("Patched {} instructions. output size {}", count,
		    result.size());
	return result;
}
