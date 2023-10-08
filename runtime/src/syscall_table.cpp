#include "syscall_table.hpp"
#include <filesystem>
#include <fstream>
#include <cassert>
#include <optional>
#include <syscall_id_list.h>
static const char *SYSCALL_TRACEPOINT_ROOT =
	"/sys/kernel/tracing/events/syscalls";

namespace bpftime
{

namespace internal
{
static syscall_id_pair generate_syscall_id_table()
{
	syscall_name_to_id_table ret1;
	syscall_id_to_name_table ret2;
	std::istringstream ss(table);
	while (ss) {
		std::string name;
		int id;
		ss >> name >> id;
		ret1[name] = id;
		ret2[id] = name;
	}
	return { ret1, ret2 };
}
} // namespace internal
const syscall_id_pair &get_global_syscall_id_table()
{
	static std::optional<syscall_id_pair> value;
	if (!value)
		value = internal::generate_syscall_id_table();
	return value.value();
}
const syscall_tracepoint_table &get_global_syscall_tracepoint_table()
{
	static std::optional<syscall_tracepoint_table> value;
	if (!value)
		value = create_syscall_tracepoint_table();
	return value.value();
}
syscall_tracepoint_table create_syscall_tracepoint_table()
{
	syscall_tracepoint_table result;
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
