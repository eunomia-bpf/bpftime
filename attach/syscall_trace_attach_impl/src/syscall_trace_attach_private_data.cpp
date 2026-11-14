#include "spdlog/spdlog.h"
#include "syscall_table.hpp"
#include <cerrno>
#include <string>
#include <syscall_trace_attach_private_data.hpp>
using namespace bpftime::attach;

int32_t syscall_trace_attach_private_data::synthetic_id_for_event(
	const std::string_view &event)
{
	if (event == "raw_syscalls/sys_enter")
		return synthetic_id_base;
	if (event == "raw_syscalls/sys_exit")
		return synthetic_id_base + 1;
	constexpr std::string_view enter_prefix = "syscalls/sys_enter_";
	constexpr std::string_view exit_prefix = "syscalls/sys_exit_";
	bool enter = event.starts_with(enter_prefix);
	if (!enter && !event.starts_with(exit_prefix))
		return 0;
	auto name = event.substr(enter ? enter_prefix.size() : exit_prefix.size());
	const auto &names = std::get<0>(get_global_syscall_id_table());
	auto found = names.find(std::string(name));
	if (name.empty() || found == names.end() || found->second < 0 ||
	    found->second >= (INT32_MAX - synthetic_id_base) / 2 - 1)
		return 0;
	return synthetic_id_base + 2 * (found->second + 1) + (enter ? 0 : 1);
}

int syscall_trace_attach_private_data::initialize_from_string(
	const std::string_view &sv)
{
	int tp_id = std::stoi(std::string(sv));
	if (tp_id >= synthetic_id_base) {
		int offset = tp_id - synthetic_id_base;
		int nr = offset / 2 - 1;
		if (nr >= 0 && !std::get<1>(get_global_syscall_id_table()).contains(nr))
			return -EINVAL;
		sys_nr = nr;
		is_enter = offset % 2 == 0;
		return 0;
	}
	SPDLOG_DEBUG(
		"Initializing syscall trace attach private data from tp_id {}",
		tp_id);
	auto &table = get_global_syscall_tracepoint_name_table();
	if (auto itr = table.find(tp_id); itr != table.end()) {
		auto &name = itr->second;
		SPDLOG_DEBUG(
			"Syscall tracepoint name of tracepoint id {} is {}",
			tp_id, name);
		if (name == GLOBAL_SYS_ENTER_NAME) {
			SPDLOG_DEBUG("Processing global sys enter");
			is_enter = true;
			sys_nr = -1;
			return 0;
		} else if (name == GLOBAL_SYS_EXIT_NAME) {
			SPDLOG_DEBUG("Processing global sys exit");
			is_enter = false;
			sys_nr = -1;
            return 0;
		} else {
			auto &tp_name_to_sys_nr =
				std::get<0>(get_global_syscall_id_table());
			std::string syscall_name;
			bool is_enter;
			if (name.starts_with("sys_enter_")) {
				syscall_name = name.substr(10);
				is_enter = true;
			} else if (name.starts_with("sys_exit_")) {
				syscall_name = name.substr(9);
				is_enter = false;
			}
			if (auto itr2 = tp_name_to_sys_nr.find(syscall_name);
			    itr2 != tp_name_to_sys_nr.end()) {
				sys_nr = itr2->second;
				this->is_enter = is_enter;
				SPDLOG_DEBUG(
					"Result: sys_nr = {}, is_enter = {}",
					sys_nr, this->is_enter);
				return 0;
			} else {
				SPDLOG_ERROR(
					"Unable to lookup sys nr for syscall tracepoint {}, syscall name {}",
					name, syscall_name);
				return -EEXIST;
			}
		}
	} else {
		SPDLOG_ERROR("Unable to find tp id {} in syscall tracepoints",
			     tp_id);
		return -EEXIST;
	}
}
