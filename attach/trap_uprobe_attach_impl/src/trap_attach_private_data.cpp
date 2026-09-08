/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "trap_attach_private_data.hpp"
#include "trap_attach_utils.hpp"
#include <cerrno>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>

using namespace bpftime::attach::trap;

int trap_attach_private_data::initialize_from_string(const std::string_view &sv)
{
	SPDLOG_DEBUG("Resolving trap attach private data from string {}", sv);
	try {
		if (sv.find(':') == std::string_view::npos) {
			addr = std::stoull(std::string(sv));
			module_name.clear();
			SPDLOG_DEBUG("Resolved address {:x} from string {}",
				     addr, sv);
			return 0;
		}
		auto pos = sv.find_last_of(':');
		if (pos == sv.length() - 1) {
			SPDLOG_ERROR(
				"Unable to parse `{}`, offset part cannot be empty",
				sv);
			return -EINVAL;
		}
		auto module_part = sv.substr(0, pos);
		auto offset_part = std::string(sv.substr(pos + 1));
		auto resolved_module = resolve_mapped_module_path(module_part);
		if (!resolved_module) {
			SPDLOG_ERROR("Unable to resolve mapped module path `{}`",
				     module_part);
			return -ENOENT;
		}
		uintptr_t offset = std::stoull(offset_part);
		addr = (uintptr_t)resolve_function_addr_by_module_offset(
			*resolved_module, offset);
		module_name = std::move(*resolved_module);
		SPDLOG_DEBUG("Resolved address {:x} from string {}", addr, sv);
		return 0;
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("Unable to parse attach target `{}`: {}", sv,
			     ex.what());
		return -EINVAL;
	}
}

std::string trap_attach_private_data::to_string() const
{
	if (module_name.empty())
		return std::to_string(addr);
	return module_name + ":" + std::to_string(addr);
}
