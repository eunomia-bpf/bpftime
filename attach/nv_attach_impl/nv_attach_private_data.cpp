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
	if (!func_names.empty()) {
		oss << "func_names=";
		for (size_t i = 0; i < func_names.size(); i++) {
			if (i)
				oss << ",";
			oss << func_names[i];
		}
		oss << " ";
	}
	oss << "comm_shared_mem=" << std::hex << comm_shared_mem << " ";
	return oss.str();
};

int nv_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	std::string input{ sv };
	std::stringstream ss(input);
	std::string item;
	while (std::getline(ss, item, ',')) {
		if (!item.empty())
			func_names.push_back(item);
	}
	if (!func_names.empty())
		code_addr_or_func_name = func_names.front();
	else
		code_addr_or_func_name = input;
	return 0;
}
