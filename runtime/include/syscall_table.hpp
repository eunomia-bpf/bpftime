#ifndef _SYSCALL_TABLE_HPP
#define _SYSCALL_TABLE_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
namespace bpftime
{
using syscall_tracepoint_table = std::unordered_map<int32_t, std::string>;
using syscall_id_to_name_table = std::unordered_map<int32_t, std::string>;
using syscall_name_to_id_table = std::unordered_map<std::string, int32_t>;
using syscall_id_pair =
	std::pair<syscall_name_to_id_table, syscall_id_to_name_table>;
// Create a mapping of id -> syscall tracepoint name using stuff at
// /sys/kernel/tracing/events/syscalls/*/id
syscall_tracepoint_table create_syscall_tracepoint_table();
// Get the global singleton syscall tracepoint table
const syscall_tracepoint_table &get_global_syscall_tracepoint_table();
// Get the name mapping of syscall name <-> syscall id
const syscall_id_pair &get_global_syscall_id_table();

constexpr const char *GLOBAL_SYS_ENTER_NAME = "*SYS_ENTER";
constexpr const char *GLOBAL_SYS_EXIT_NAME = "*SYS_EXIT";

} // namespace bpftime

#endif
