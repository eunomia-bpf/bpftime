#include "nv_attach_private_data.hpp"
#include <iterator>
#include <spdlog/fmt/fmt.h>
#include <string>

using namespace bpftime;
using namespace attach;
namespace fmt_lib = spdlog::fmt_lib;

std::string nv_attach_private_data::to_string() const
{
	fmt_lib::memory_buffer buffer;
	fmt_lib::format_to(std::back_inserter(buffer),
			   "nv_attach_private_data: ");
	if (std::holds_alternative<uintptr_t>(code_addr_or_func_name)) {
		fmt_lib::format_to(std::back_inserter(buffer),
				   "code_addr={:x} ",
				   std::get<uintptr_t>(code_addr_or_func_name));
	} else {
		fmt_lib::format_to(std::back_inserter(buffer), "func_name={} ",
				   std::get<std::string>(
					   code_addr_or_func_name));
	}
	if (!func_names.empty()) {
		fmt_lib::format_to(std::back_inserter(buffer), "func_names=");
		for (size_t i = 0; i < func_names.size(); i++) {
			if (i)
				fmt_lib::format_to(std::back_inserter(buffer),
						   ",");
			fmt_lib::format_to(std::back_inserter(buffer), "{}",
					   func_names[i]);
		}
		fmt_lib::format_to(std::back_inserter(buffer), " ");
	}
	fmt_lib::format_to(std::back_inserter(buffer), "comm_shared_mem={:x} ",
			   comm_shared_mem);
	return fmt_lib::to_string(buffer);
};

int nv_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	std::string input{ sv };
	size_t start = 0;
	while (start < sv.size()) {
		const size_t end = sv.find(',', start);
		const auto item = sv.substr(start, end - start);
		if (!item.empty())
			func_names.emplace_back(item);
		if (end == std::string_view::npos)
			break;
		start = end + 1;
	}
	if (!func_names.empty())
		code_addr_or_func_name = func_names.front();
	else
		code_addr_or_func_name = input;
	return 0;
}
