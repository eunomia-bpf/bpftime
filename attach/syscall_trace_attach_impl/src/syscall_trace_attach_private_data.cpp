#include "spdlog/spdlog.h"
#include "syscall_table.hpp"
#include <charconv>
#include <cerrno>
#include <string>
#include <syscall_trace_attach_private_data.hpp>
using namespace bpftime::attach;

int syscall_trace_attach_private_data::initialize_from_string(
	const std::string_view &sv)
{
	constexpr std::string_view x86_syscall_prefix = "__x64_sys_";
	constexpr std::string_view arm64_syscall_prefix = "__arm64_sys_";
	constexpr std::string_view kernel_do_prefix = "do_";
	std::string_view syscall_name;
	if (sv.starts_with(x86_syscall_prefix))
		syscall_name = sv.substr(x86_syscall_prefix.size());
	else if (sv.starts_with(arm64_syscall_prefix))
		syscall_name = sv.substr(arm64_syscall_prefix.size());
	else if (sv.starts_with(kernel_do_prefix)) {
		syscall_name = sv.substr(kernel_do_prefix.size());
		if (syscall_name == "mmap")
			kprobe_abi = syscall_kprobe_abi::do_mmap;
		else if (syscall_name == "munmap")
			kprobe_abi = syscall_kprobe_abi::do_munmap;
	}
	if (!syscall_name.empty()) {
		auto &syscall_ids =
			std::get<0>(get_global_syscall_id_table());
		if (auto itr = syscall_ids.find(std::string(syscall_name));
		    itr != syscall_ids.end()) {
			sys_nr = itr->second;
			is_enter = true;
			return 0;
		}
		SPDLOG_ERROR("Unable to lookup syscall {}", syscall_name);
		return -ENOENT;
	}
	int tp_id = 0;
	auto [end, error] =
		std::from_chars(sv.data(), sv.data() + sv.size(), tp_id);
	if (error != std::errc() || end != sv.data() + sv.size()) {
		SPDLOG_ERROR("Invalid syscall trace private data {}", sv);
		return -EINVAL;
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
