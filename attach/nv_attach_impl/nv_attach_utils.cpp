#include "nv_attach_utils.hpp"
#include "trampoline_ptx.h"
#include <spdlog/spdlog.h>
namespace bpftime
{
namespace attach
{
std::string get_defaul_trampoline_ptx()
{
	return TRAMPOLINE_PTX;
}
std::string wrap_ptx_with_trampoline(std::string input)
{
	return get_defaul_trampoline_ptx() + input;
}
std::string patch_helper_names_and_header(std::string result)
{
	const std::string to_replace_names[][2] = {
		{ "_bpf_helper_ext_0001", "_bpf_helper_ext_0001_dup" },
		{ "_bpf_helper_ext_0002", "_bpf_helper_ext_0002_dup" },
		{ "_bpf_helper_ext_0003", "_bpf_helper_ext_0003_dup" },
		{ "_bpf_helper_ext_0006", "_bpf_helper_ext_0006_dup" },

	};
	const std::string version_headers[] = {
		".version 3.2\n.target sm_90\n.address_size 64\n",
		".version 5.0\n.target sm_90\n.address_size 64\n"
	};
	for (const auto &entry : to_replace_names) {
		auto idx = result.find(entry[0]);
		if (idx != result.npos) {
			result = result.replace(idx, entry[0].size(), entry[1]);
		}
	}
	for (const auto &header : version_headers) {
		auto idx = result.find(header);
		SPDLOG_INFO("Version header ({}) index: {}", header, idx);
		if (idx != result.npos) {
			result = result.replace(idx, header.size(), "");
		}
	}
	return result;
}
std::string patch_main_from_func_to_entry(std::string result)
{
	const std::string entry_func = ".visible .func bpf_main";

	auto idx = result.find(entry_func);
	SPDLOG_INFO("entry_func ({}) index {}", entry_func, idx);

	if (idx != result.npos) {
		result = result.replace(idx, entry_func.size(),
					".visible .entry bpf_main");
	}
	return result;
}

} // namespace attach
} // namespace bpftime
