#include <frida_attach_private_data.hpp>
#include <spdlog/spdlog.h>
#include <string>
using namespace bpftime::attach;

int frida_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	// Resolve function address from the string
	addr = std::stoul(std::string(sv));
	SPDLOG_DEBUG("Resolved address {} from string {}", addr, sv);
	return 0;
}
