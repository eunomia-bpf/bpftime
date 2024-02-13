#include "syscall_table.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <syscall_trace_attach_impl.hpp>
using namespace bpftime::attach;

extern "C" int64_t _bpftime_dummy_syscall(int64_t, int64_t, int64_t, int64_t,
					  int64_t, int64_t, int64_t)
{
	return 0;
}

TEST_CASE("Test syscall dispatch - global")
{
	syscall_trace_attach_impl attacher;
	attacher.set_original_syscall_function(_bpftime_dummy_syscall);

	syscall_trace_attach_private_data data;
	bool set = false;
	auto &tp_table = get_global_syscall_tracepoint_name_table();

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
	bool invoked = false;
	int id1 = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t, uint64_t *) -> int {
			invoked = true;
			auto &ctx = *(trace_event_raw_sys_enter *)p;
			REQUIRE(ctx.args[0] == 0xabcd);
			return 0;
		},
		data, ATTACH_SYSCALL_TRACE);
	REQUIRE(id1 >= 0);
	attacher.dispatch_syscall(11, 0xabcd, 0, 0, 0, 0, 0);
	REQUIRE(invoked == true);

	REQUIRE(attacher.detach_by_id(id1) == 0);
	invoked = false;
	attacher.dispatch_syscall(11, 0xabcd, 0, 0, 0, 0, 0);
	REQUIRE(invoked == false);
}

TEST_CASE("Test syscall dispatch - multiple")
{
	syscall_trace_attach_impl attacher;
	attacher.set_original_syscall_function(_bpftime_dummy_syscall);

	syscall_trace_attach_private_data data_global_enter, data_read_enter,
		data_write_enter, data_read_exit;
	int set_cnt = 0;
	auto &tp_table = get_global_syscall_tracepoint_name_table();

	auto &sys_name_to_nr = std::get<0>(get_global_syscall_id_table());
	REQUIRE(sys_name_to_nr.contains("read"));
	REQUIRE(sys_name_to_nr.contains("write"));

	for (auto &[k, v] : tp_table) {
		if (v == GLOBAL_SYS_ENTER_NAME) {
			REQUIRE(data_global_enter.initialize_from_string(
					std::to_string(k)) == 0);
			REQUIRE(data_global_enter.is_enter == true);
			REQUIRE(data_global_enter.sys_nr == -1);
			set_cnt++;
		} else if (v == "sys_enter_read") {
			REQUIRE(data_read_enter.initialize_from_string(
					std::to_string(k)) == 0);
			REQUIRE(data_read_enter.is_enter == true);
			REQUIRE(data_read_enter.sys_nr ==
				sys_name_to_nr.at("read"));
			set_cnt++;

		} else if (v == "sys_enter_write") {
			REQUIRE(data_write_enter.initialize_from_string(
					std::to_string(k)) == 0);

			REQUIRE(data_write_enter.is_enter == true);
			REQUIRE(data_write_enter.sys_nr ==
				sys_name_to_nr.at("write"));
			set_cnt++;
		} else if (v == "sys_exit_read") {
			REQUIRE(data_read_exit.initialize_from_string(
					std::to_string(k)) == 0);

			REQUIRE(data_read_exit.is_enter == false);
			REQUIRE(data_read_exit.sys_nr ==
				sys_name_to_nr.at("read"));
			set_cnt++;
		}
	}
	REQUIRE(set_cnt == 4);
	int invoke_cnt = 0;
	int id_global_enter = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t, uint64_t *) -> int {
			invoke_cnt++;
			return 0;
		},
		data_global_enter, ATTACH_SYSCALL_TRACE);
	REQUIRE(id_global_enter >= 0);

	int id_read_enter = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t, uint64_t *) -> int {
			invoke_cnt++;
			return 0;
		},
		data_read_enter, ATTACH_SYSCALL_TRACE);
	REQUIRE(id_read_enter >= 0);

	int id_write_enter = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t, uint64_t *) -> int {
			invoke_cnt++;
			return 0;
		},
		data_write_enter, ATTACH_SYSCALL_TRACE);
	REQUIRE(id_write_enter >= 0);

	int id_read_exit = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t, uint64_t *) -> int {
			invoke_cnt++;
			return 0;
		},
		data_read_exit, ATTACH_SYSCALL_TRACE);
	REQUIRE(id_read_exit >= 0);

	SECTION("call ton read")
	{
		// Dispatch a call to read
		attacher.dispatch_syscall(sys_name_to_nr.at("read"), 0, 0, 0, 0,
					  0, 0);
		REQUIRE(invoke_cnt == 3);
	}
	SECTION("call to write")
	{
		// Dispatch a call to write
		attacher.dispatch_syscall(sys_name_to_nr.at("write"), 0, 0, 0,
					  0, 0, 0);
		REQUIRE(invoke_cnt == 2);
	}
	SECTION("call to fork")
	{
		// Dispatch a call to fork
		attacher.dispatch_syscall(sys_name_to_nr.at("fork"), 0, 0, 0, 0,
					  0, 0);
		REQUIRE(invoke_cnt == 1);
	}

	REQUIRE(attacher.detach_by_id(id_global_enter) == 0);
	invoke_cnt = 0;
	attacher.dispatch_syscall(sys_name_to_nr.at("read"), 0, 0, 0, 0, 0, 0);
	REQUIRE(invoke_cnt == 2);

	REQUIRE(attacher.detach_by_id(id_read_enter) == 0);
	REQUIRE(attacher.detach_by_id(id_read_exit) == 0);
	invoke_cnt = 0;
	attacher.dispatch_syscall(sys_name_to_nr.at("read"), 0, 0, 0, 0, 0, 0);
	REQUIRE(invoke_cnt == 0);

	invoke_cnt = 0;
	attacher.dispatch_syscall(sys_name_to_nr.at("write"), 0, 0, 0, 0, 0, 0);
	REQUIRE(invoke_cnt == 1);

	REQUIRE(attacher.detach_by_id(id_write_enter) == 0);
	invoke_cnt = 0;
	attacher.dispatch_syscall(sys_name_to_nr.at("write"), 0, 0, 0, 0, 0, 0);
	REQUIRE(invoke_cnt == 0);
}
