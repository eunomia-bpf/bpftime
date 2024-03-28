/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "syscall_table.hpp"
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <syscall_id_list.h>
static const char *SYSCALL_TRACEPOINT_ROOT =
	"/sys/kernel/tracing/events/syscalls";
static const char *TRACEPOINT_ROOT = "/sys/kernel/tracing/events";
static bpftime::attach::syscall_id_pair generate_syscall_id_table()
{
	// Some syscalls have different name between tracepoint name and syscall
	// name Here maintains a mapping from syscall name to tracepoint name
	// Syscall name comes from  /usr/include/asm/unistd_64.h, while
	// tracepoint name comes from /sys/kernel/tracing/events/syscalls/ Don't
	// use static variable. Avoid global variable initializing issues.
	const std::map<std::string, std::string> syscall_name_patch{
		{ "umount2", "umount" }
	};
	bpftime::attach::syscall_name_to_id_table ret1;
	bpftime::attach::syscall_id_to_name_table ret2;
	std::istringstream ss(table);
	while (ss) {
		std::string name;
		int id;
		ss >> name >> id;
		if (auto itr = syscall_name_patch.find(name);
		    itr != syscall_name_patch.end()) {
			SPDLOG_DEBUG("Patched syscall name {} to {}", name,
				     itr->second);
			name = itr->second;
		}
		ret1[name] = id;
		ret2[id] = name;
	}
	return { ret1, ret2 };
}
namespace bpftime
{

namespace attach
{
const syscall_id_pair &get_global_syscall_id_table()
{
	static std::optional<syscall_id_pair> value;
	if (!value)
		value = generate_syscall_id_table();
	return value.value();
}

const syscall_tracepoint_table &get_global_syscall_tracepoint_name_table()
{
	static std::optional<syscall_tracepoint_table> value;
	if (!value)
		value = create_syscall_tracepoint_id_table();
	return value.value();
}

syscall_tracepoint_table create_syscall_tracepoint_id_table()
{
	syscall_tracepoint_table result;
	const auto read_id = [&](std::filesystem::path tp_dir) -> int32_t {
		const auto &id_file = tp_dir.append("id");
		SPDLOG_TRACE("Reading tracepoint id from {}", id_file.string());
		std::ifstream id_ifs(id_file);
		if (!id_ifs.is_open()) {
			SPDLOG_ERROR("Unable to open & read {}",
				     id_file.c_str());
			throw std::runtime_error("Unable to open id file");
		}
		int32_t id;
		id_ifs >> id;
		return id;
	};
	for (const auto &entry :
	     std::filesystem::directory_iterator(SYSCALL_TRACEPOINT_ROOT)) {
		if (entry.is_directory()) {
			auto curr_path = entry.path();
			auto tp_name = curr_path.filename();
			result[read_id(curr_path)] = tp_name;
		}
	}
	result[read_id(std::filesystem::path(TRACEPOINT_ROOT)
			       .append("raw_syscalls")
			       .append("sys_enter"))] = GLOBAL_SYS_ENTER_NAME;
	result[read_id(std::filesystem::path(TRACEPOINT_ROOT)
			       .append("raw_syscalls")
			       .append("sys_exit"))] = GLOBAL_SYS_EXIT_NAME;

	return result;
}
} // namespace attach
} // namespace bpftime
