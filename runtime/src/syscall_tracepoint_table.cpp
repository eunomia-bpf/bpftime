#include "syscall_tracepoint_table.hpp"
#include <filesystem>
#include <fstream>
#include <cassert>
#include <optional>
static const char *SYSCALL_TRACEPOINT_ROOT =
	"/sys/kernel/tracing/events/syscalls";

namespace bpftime
{

const std::unordered_map<int32_t, std::string> &
get_global_syscall_tracepoint_table()
{
	static std::optional<std::unordered_map<int32_t, std::string> > value;
	if (!value)
		value = create_syscall_tracepoint_table();
	return value.value();
}
std::unordered_map<int32_t, std::string> create_syscall_tracepoint_table()
{
	std::unordered_map<int32_t, std::string> result;
	for (const auto &entry :
	     std::filesystem::directory_iterator(SYSCALL_TRACEPOINT_ROOT)) {
		if (entry.is_directory()) {
			auto curr_path = entry.path();
			auto tp_name = curr_path.filename();
			const auto &id_file = curr_path.append("id");
			std::ifstream id_ifs(id_file);
			assert(id_ifs.is_open());
			int32_t id;
			id_ifs >> id;
			result[id] = tp_name;
		}
	}
	return result;
}
} // namespace bpftime
