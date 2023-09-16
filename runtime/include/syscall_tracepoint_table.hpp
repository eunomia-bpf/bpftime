#ifndef _SYSCALL_TRACEPOINT_TABLE_HPP
#define _SYSCALL_TRACEPOINT_TABLE_HPP

#include <string>
#include <unordered_map>
#include <cinttypes>
namespace bpftime
{
// Create a mapping of id -> syscall tracepoint name using stuff at
// /sys/kernel/tracing/events/syscalls/*/id

std::unordered_map<int32_t, std::string> create_syscall_tracepoint_table();
// Get the global singleton syscall tracepoint table
const std::unordered_map<int32_t, std::string> &
get_global_syscall_tracepoint_table();
} // namespace bpftime

#endif
