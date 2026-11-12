#include "frida_attach_utils.hpp"
#include <cerrno>
#include <frida_attach_private_data.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>
using namespace bpftime::attach;

int frida_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	SPDLOG_DEBUG("Resolving frida attach private data from string {}", sv);
	if (sv.find(':') == std::string_view::npos) {
		// Resolve function address from the string
		addr = std::stoul(std::string(sv));
		SPDLOG_DEBUG("Resolved address {} from string {}", addr, sv);
	} else {
		auto pos = sv.find_last_of(':');
		if (pos == sv.length() - 1) {
			SPDLOG_ERROR(
				"Unable to parse `{}`, offset part cannot be empty",
				sv);
			return -EINVAL;
		}
		auto module_part = sv.substr(0, pos);
		auto offset_part = std::string(sv.substr(pos + 1));
		SPDLOG_DEBUG("Module part is `{}`, offset part is `{}`",
			     module_part, offset_part);
		auto resolved_module = resolve_mapped_module_path(module_part);
		if (!resolved_module) {
			SPDLOG_ERROR("Unable to resolve mapped module path `{}`",
				     module_part);
			return -ENOENT;
		}
		addr = (uintptr_t)resolve_function_addr_by_module_offset(
			*resolved_module, std::stoul(offset_part));
		SPDLOG_DEBUG("Resolved address: {:x} from string {}", addr, sv);
		this->module_name = std::move(*resolved_module);
	}

	return 0;
}
