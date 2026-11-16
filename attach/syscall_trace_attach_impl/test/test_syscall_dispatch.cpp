#include "syscall_table.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <syscall_trace_attach_impl.hpp>
using namespace bpftime::attach;

extern "C" int64_t _bpftime_dummy_syscall(int64_t, int64_t, int64_t, int64_t,
					  int64_t, int64_t, int64_t, int64_t,
					  int64_t, int64_t)
{
	return 0;
}

static int syscall_call_count;
extern "C" int64_t _bpftime_counting_syscall(int64_t, int64_t, int64_t, int64_t,
					     int64_t, int64_t, int64_t, int64_t,
					     int64_t, int64_t)
{
	return ++syscall_call_count;
}

#if defined(__x86_64__)
struct test_pt_regs {
	uint64_t r15, r14, r13, r12, bp, bx, r11, r10, r9, r8, ax, cx, dx,
		si, di, orig_ax, ip, cs, flags, sp, ss;
};

TEST_CASE("do_mmap kprobe receives kernel function arguments")
{
	auto &syscalls = std::get<0>(get_global_syscall_id_table());
	REQUIRE(syscalls.contains("mmap"));
	syscall_trace_attach_impl attacher;
	attacher.set_original_syscall_function(_bpftime_dummy_syscall);
	syscall_trace_attach_private_data data;
	REQUIRE(data.initialize_from_string("do_mmap") == 0);
	test_pt_regs observed = {};
	REQUIRE(attacher.create_attach_with_ebpf_callback(
			[&](const void *p, size_t size, uint64_t *) -> int {
				if (size == sizeof(observed))
					observed = *static_cast<const test_pt_regs *>(p);
				return 0;
			},
			data, ATTACH_SYSCALL_KPROBE) >= 0);
	attacher.dispatch_syscall(syscalls.at("mmap"), 0x1000, 0x2000, 3,
				  0x22, -1, 0, 0xabc, 0xdef, 0x123);
	REQUIRE(observed.di == 0);
	REQUIRE(observed.si == 0x1000);
	REQUIRE(observed.dx == 0x2000);
	REQUIRE(observed.cx == 3);
	REQUIRE(observed.r8 == 0x22);
	REQUIRE(observed.ip == 0xabc);
	REQUIRE(observed.sp == 0xdef);
	REQUIRE(observed.bp == 0x123);
}
#endif

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

TEST_CASE("A failing exit callback is contained")
{
	syscall_trace_attach_private_data data;
	data.sys_nr = 11;
	data.is_enter = false;
	syscall_trace_attach_impl attacher;
	attacher.set_original_syscall_function(_bpftime_counting_syscall);
	REQUIRE(attacher.create_attach_with_ebpf_callback(
			[&](const void *, size_t, uint64_t *) -> int {
				throw std::runtime_error(
					"test callback failure");
			},
			data, ATTACH_SYSCALL_TRACE) >= 0);
	int later_callback_count = 0;
	data.sys_nr = -1;
	REQUIRE(attacher.create_attach_with_ebpf_callback(
			[&](const void *, size_t, uint64_t *) -> int {
				return ++later_callback_count;
			},
			data, ATTACH_SYSCALL_TRACE) >= 0);
	syscall_call_count = 0;
	REQUIRE(attacher.dispatch_syscall(11, 0, 0, 0, 0, 0, 0) == 1);
	REQUIRE(syscall_call_count == 1);
	REQUIRE(later_callback_count == 1);
	REQUIRE(!curr_thread_override_return_callback.has_value());
}

TEST_CASE("Scheduler lifecycle tracepoints are dispatched")
{
	REQUIRE(offsetof(trace_event_raw_sched_process_fork, parent_pid) == 24);
	REQUIRE(offsetof(trace_event_raw_sched_process_fork, child_pid) == 44);
	auto tracepoint_id = [](const char *name) {
		for (const auto &[id, mapped_name] :
		     get_global_syscall_tracepoint_name_table()) {
			if (mapped_name == name)
				return id;
		}
		return -1;
	};
	auto make_data = [&](const char *name) {
		syscall_trace_attach_private_data data;
		int id = tracepoint_id(name);
		REQUIRE(id >= 0);
		REQUIRE(data.initialize_from_string(std::to_string(id)) == 0);
		return data;
	};

	syscall_trace_attach_impl attacher;
	attacher.set_original_syscall_function(_bpftime_counting_syscall);
	bool fork_invoked = false;
	bool exec_invoked = false;
	bool exit_invoked = false;
	size_t fork_ctx_size = 0;
	size_t exec_ctx_size = 0;
	size_t exit_ctx_size = 0;
	int parent_pid = 0;
	int child_pid = 0;
	auto fork_data = make_data(SCHED_PROCESS_FORK_NAME);
	int fork_id = attacher.create_attach_with_ebpf_callback(
		[&](const void *p, size_t size, uint64_t *) -> int {
			auto &ctx = *static_cast<
				const trace_event_raw_sched_process_fork *>(p);
			fork_ctx_size = size;
			parent_pid = ctx.parent_pid;
			child_pid = ctx.child_pid;
			fork_invoked = true;
			return 0;
		},
		fork_data, ATTACH_SYSCALL_TRACE);
	REQUIRE(fork_id >= 0);
	auto exec_data = make_data(SCHED_PROCESS_EXEC_NAME);
	int exec_id = attacher.create_attach_with_ebpf_callback(
		[&](const void *, size_t size, uint64_t *) -> int {
			exec_ctx_size = size;
			exec_invoked = true;
			return 0;
		},
		exec_data, ATTACH_SYSCALL_TRACE);
	REQUIRE(exec_id >= 0);
	auto exit_data = make_data(SCHED_PROCESS_EXIT_NAME);
	int exit_id = attacher.create_attach_with_ebpf_callback(
		[&](const void *, size_t size, uint64_t *) -> int {
			exit_ctx_size = size;
			exit_invoked = true;
			return 0;
		},
		exit_data, ATTACH_SYSCALL_TRACE);
	REQUIRE(exit_id >= 0);

	auto &syscalls = std::get<0>(get_global_syscall_id_table());
	REQUIRE(syscalls.contains("clone"));
	syscall_call_count = 0;
	REQUIRE(attacher.dispatch_syscall(syscalls.at("clone"), 0, 0, 0, 0, 0,
					  0) == 1);
	attacher.dispatch_process_exec();
	attacher.dispatch_process_exit();
	REQUIRE(fork_invoked);
	REQUIRE(exec_invoked);
	REQUIRE(exit_invoked);
	REQUIRE(fork_ctx_size == sizeof(trace_event_raw_sched_process_fork));
	REQUIRE(exec_ctx_size == sizeof(trace_entry));
	REQUIRE(exit_ctx_size == sizeof(trace_entry));
	REQUIRE(parent_pid > 0);
	REQUIRE(child_pid == 1);

	REQUIRE(attacher.detach_by_id(fork_id) == 0);
	REQUIRE(attacher.detach_by_id(exec_id) == 0);
	REQUIRE(attacher.detach_by_id(exit_id) == 0);
}
