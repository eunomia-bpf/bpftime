#include "nv_attach_private_data.hpp"
#include <ios>
#include <ostream>
#include <sstream>
#include <string>

using namespace bpftime;
using namespace attach;

std::string nv_attach_private_data::to_string() const
{
	std::ostringstream oss{};
	oss << "nv_attach_private_data: ";
	if (std::holds_alternative<uintptr_t>(code_addr_or_func_name))
		oss << "code_addr=" << std::hex
		    << std::get<uintptr_t>(code_addr_or_func_name) << " ";
	else {
		oss << "func_name=" << std::hex
		    << std::get<std::string>(code_addr_or_func_name) << " ";
	}
	oss << "comm_shared_mem=" << std::hex << comm_shared_mem << " ";
	oss << "trampoline_ptx=" << trampoline_ptx << std::endl;
	return oss.str();
};

int nv_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	this->code_addr_or_func_name = std::string(sv);

	return 0;
}
