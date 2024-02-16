#include "syscall_table.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include <catch2/catch_test_macros.hpp>
#include <syscall_trace_attach_impl.hpp>
#include <string>
using namespace bpftime::attach;

TEST_CASE("Test resolving private data")
{
	auto &tp_table = get_global_syscall_tracepoint_name_table();

	syscall_trace_attach_private_data data;
	bool set = false;
	SECTION("global enter")
	{
		for (auto &[k, v] : tp_table) {
			if (v == GLOBAL_SYS_ENTER_NAME) {
				REQUIRE(data.initialize_from_string(
						std::to_string(k)) == 0);
				set = true;
				break;
			}
		}
		REQUIRE(set);
		REQUIRE(data.is_enter == true);
		REQUIRE(data.sys_nr == -1);
	}
	SECTION("global exit")
	{
		for (auto &[k, v] : tp_table) {
			if (v == GLOBAL_SYS_EXIT_NAME) {
				REQUIRE(data.initialize_from_string(
						std::to_string(k)) == 0);
				set = true;
				break;
			}
		}
		REQUIRE(set);
		REQUIRE(data.is_enter == false);
		REQUIRE(data.sys_nr == -1);
	}
	SECTION("fork enter")
	{
		auto &sys_name_to_nr =
			std::get<0>(get_global_syscall_id_table());
		REQUIRE(sys_name_to_nr.contains("fork"));

		for (auto &[k, v] : tp_table) {
			if (v == "sys_enter_fork") {
				REQUIRE(data.initialize_from_string(
						std::to_string(k)) == 0);
				set = true;
				break;
			}
		}
		REQUIRE(set);
		REQUIRE(data.is_enter == true);
		REQUIRE(data.sys_nr == sys_name_to_nr.at("fork"));
	}
}
