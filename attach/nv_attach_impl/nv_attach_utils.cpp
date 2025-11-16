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
