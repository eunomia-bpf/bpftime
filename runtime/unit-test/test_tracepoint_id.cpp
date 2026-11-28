
#if __linux__
#include "syscall_server_utils.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include "syscall_trace_attach_impl.hpp"
#include "platform_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <boost/scope_exit.hpp>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <atomic>
#include <barrier>
#include <thread>
#include <vector>
#include <cstdarg>

// Control only the real-id probe; the provider still uses RTLD_NEXT and
// reads/closes a genuine descriptor. All other symbol lookups use libc.
static bool test_tracepoint_probe = false;
static int test_tracepoint_fd = -1;
extern "C" void *__real_dlsym(void *, const char *) noexcept;
static int tracepoint_test_openat(int dirfd, const char *path, int flags, ...)
{
	if (test_tracepoint_probe) {
		if (test_tracepoint_fd < 0) { errno = ENOENT; return -1; }
		lseek(test_tracepoint_fd, 0, SEEK_SET);
		return dup(test_tracepoint_fd);
	}
	using fn = int (*)(int, const char *, int, ...);
	mode_t mode = 0;
	if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
		va_list args;
		va_start(args, flags);
		mode = va_arg(args, int);
		va_end(args);
	}
	return reinterpret_cast<fn>(__real_dlsym(RTLD_NEXT, "openat"))(dirfd, path, flags, mode);
}
extern "C" void *__wrap_dlsym(void *handle, const char *name) noexcept
{
	if (handle == RTLD_NEXT && strcmp(name, "openat") == 0)
		return reinterpret_cast<void *>(tracepoint_test_openat);
	return __real_dlsym(handle, name);
}

TEST_CASE("Tracepoint id provider preserves real IDs and resolves rootless aliases")
{
	test_tracepoint_probe = true;
	BOOST_SCOPE_EXIT_ALL(&) { test_tracepoint_probe = false; test_tracepoint_fd = -1; };
	constexpr int32_t expected = 0x40000000 + 2 * (SYS_read + 1);
	for (const char *root : { "/sys/kernel/tracing/events/", "/sys/kernel/debug/tracing/events/" }) {
		auto content = bpftime::get_synthetic_file_content(
			std::string(root) + "syscalls/sys_enter_read/id");
		REQUIRE(content.has_value());
		REQUIRE(*content == std::to_string(expected) + "\n");
		// An unknown syscall never gets a syscall-base id. It still
		// receives a path-derived fallback provider so a rootless
		// openat() succeeds; the value is exactly the id the table
		// builder derives for the same path.
		auto unknown = bpftime::get_synthetic_file_content(
			std::string(root) + "syscalls/sys_enter_nonexistent_bpftime/id");
		REQUIRE(unknown.has_value());
		const int32_t unknown_id = std::stoi(*unknown);
		REQUIRE(unknown_id <
			bpftime::attach::syscall_trace_attach_private_data::synthetic_id_base);
		REQUIRE(unknown_id == bpftime::rootless::synthetic_tracepoint_id(
			std::string(root) + "syscalls/sys_enter_nonexistent_bpftime/id"));
	}
	FILE *real_id = tmpfile();
	REQUIRE(real_id != nullptr);
	BOOST_SCOPE_EXIT_ALL(=) { fclose(real_id); };
	REQUIRE(fputs("123\n", real_id) >= 0);
	REQUIRE(fflush(real_id) == 0);
	test_tracepoint_fd = fileno(real_id);
	REQUIRE_FALSE(bpftime::get_synthetic_file_content(
		"/sys/kernel/tracing/events/syscalls/sys_enter_read/id").has_value());
}

TEST_CASE("Rootless tracepoint IDs resolve before any callback table access")
{
	using namespace bpftime::attach;
	for (bool enter : { true, false }) {
		syscall_trace_attach_private_data data;
		const int id = 0x40000000 + 2 * (SYS_read + 1) + (enter ? 0 : 1);
		REQUIRE(data.initialize_from_string(std::to_string(id)) == 0);
		REQUIRE(data.sys_nr == SYS_read);
		REQUIRE(data.is_enter == enter);
		syscall_trace_attach_impl attacher;
		attacher.set_original_syscall_function(
			+[](int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
			    int64_t, int64_t, int64_t) -> int64_t { return 0; });
		int calls = 0;
		int attached = attacher.create_attach_with_ebpf_callback(
			[&](const void *, size_t, uint64_t *) { ++calls; return 0; }, data, ATTACH_SYSCALL_TRACE);
		REQUIRE(attached >= 0);
		attacher.dispatch_syscall(SYS_read, 0, 0, 0, 0, 0, 0);
		REQUIRE(calls == 1);
		REQUIRE(attacher.detach_by_id(attached) == 0);
	}
	for (int id : { 0x40000000, 0x40000001 }) {
		syscall_trace_attach_private_data data;
		REQUIRE(data.initialize_from_string(std::to_string(id)) == 0);
		REQUIRE(data.sys_nr == -1);
		REQUIRE(data.is_enter == (id == 0x40000000));
	}
	syscall_trace_attach_private_data invalid;
	REQUIRE(invalid.initialize_from_string(std::to_string(INT_MAX)) == -EINVAL);
	for (const char *malformed : { "", "not-an-id", "2147483648",
				      "999999999999999999999999999999" })
		REQUIRE(invalid.initialize_from_string(malformed) == -EINVAL);
}
TEST_CASE("Rootless tracepoint IDs initialize concurrently")
{
	using bpftime::attach::syscall_trace_attach_private_data;
	std::barrier start(16);
	std::atomic<int> matched{0};
	std::vector<std::thread> workers;
	for (int i = 0; i < 16; ++i)
		workers.emplace_back([&] {
			start.arrive_and_wait();
			if (syscall_trace_attach_private_data::synthetic_id_for_event(
				    "syscalls/sys_enter_read") == 0x40000000 + 2 * (SYS_read + 1))
				++matched;
		});
	for (auto &worker : workers)
		worker.join();
	REQUIRE(matched == 16);
}
TEST_CASE("Rootless mocks sched and raw_syscalls tracepoint ids")
{
	// Regression: synthetic_id_for_event covers syscalls/sys_enter_*,
	// syscalls/sys_exit_* and the raw_syscalls aliases, but returns 0 for
	// ordinary tracepoints such as sched/.... Those used to get no mock
	// provider, so reading their tracefs id file returned ENOENT and an
	// unmodified profiler aborted at startup. Force the real-id probe to
	// fail, reproducing an unreadable rootless tracefs without depending on
	// the host kernel.
	test_tracepoint_probe = true;
	BOOST_SCOPE_EXIT_ALL(&) {
		test_tracepoint_probe = false;
		test_tracepoint_fd = -1;
	};
	using bpftime::rootless::synthetic_tracepoint_id;
	using bpftime::attach::syscall_trace_attach_private_data;
	constexpr int32_t base =
		syscall_trace_attach_private_data::synthetic_id_base;
	// sched/... has no syscall-base encoding and must use the path-derived
	// FNV id, identical to the value the rootless table builder derives for
	// the same path, and below the syscall-base range.
	const char *sched_events[] = {
		"sched/sched_process_fork",
		"sched/sched_process_exec",
		"sched/sched_process_exit",
	};
	for (const char *event_path : sched_events) {
		const std::string id_path =
			std::string("/sys/kernel/tracing/events/") + event_path + "/id";
		auto content = bpftime::get_synthetic_file_content(id_path);
		REQUIRE(content.has_value());
		const int32_t id = std::stoi(*content);
		REQUIRE(id == synthetic_tracepoint_id(id_path));
		REQUIRE(id < base);
		// Both tracefs mount aliases must resolve to the same ID, since the
		// rootless attach table indexes only the canonical tracefs path.
		const std::string debug_id_path =
			std::string("/sys/kernel/debug/tracing/events/") +
			event_path + "/id";
		auto debug_content =
			bpftime::get_synthetic_file_content(debug_id_path);
		REQUIRE(debug_content.has_value());
		REQUIRE(std::stoi(*debug_content) == id);
	}
	// raw_syscalls keeps its dedicated global enter/exit syscall-base id.
	const struct { const char *event; int32_t id; } raw_cases[] = {
		{ "raw_syscalls/sys_enter", base },
		{ "raw_syscalls/sys_exit", base + 1 },
	};
	for (const auto &rc : raw_cases) {
		const std::string id_path =
			std::string("/sys/kernel/tracing/events/") + rc.event + "/id";
		auto content = bpftime::get_synthetic_file_content(id_path);
		REQUIRE(content.has_value());
		REQUIRE(std::stoi(*content) == rc.id);
	}
}

TEST_CASE("Possible CPUs list one entry per vCPU slot")
{
	REQUIRE(setenv("BPFTIME_VCPU_RATIO", "2", 1) == 0);
	BOOST_SCOPE_EXIT_ALL(&) { unsetenv("BPFTIME_VCPU_RATIO"); };
	const int slots = bpftime_vcpu_slots();
	REQUIRE(slots == std::min(2 * bpftime_possible_cpus() + 1, 1024));
	auto content = bpftime::get_synthetic_file_content(
		"/sys/devices/system/cpu/possible");
	REQUIRE(content.has_value());
	REQUIRE(*content == "0-" + std::to_string(slots - 1) + "\n");
	// Kernel maps keep one column per real CPU.
	REQUIRE(setenv("BPFTIME_RUN_WITH_KERNEL", "1", 1) == 0);
	BOOST_SCOPE_EXIT_ALL(&) { unsetenv("BPFTIME_RUN_WITH_KERNEL"); };
	REQUIRE_FALSE(bpftime::get_synthetic_file_content(
		"/sys/devices/system/cpu/possible").has_value());
}
#endif
