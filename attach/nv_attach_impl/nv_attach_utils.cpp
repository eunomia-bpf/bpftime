#include "nv_attach_utils.hpp"
#include "trampoline_ptx.h"
#include <iomanip>
#include <spdlog/spdlog.h>
#include <openssl/sha.h>
#include <sstream>
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
std::string sha256(const void *data, size_t length)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	SHA256((unsigned char *)data, length, hash);

	std::stringstream ss;
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		ss << std::hex << std::setw(2) << std::setfill('0')
		   << (int)hash[i];
	}
	return ss.str();
}
} // namespace attach
} // namespace bpftime
