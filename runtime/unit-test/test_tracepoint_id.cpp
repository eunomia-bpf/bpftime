
#if __linux__
#include "syscall_server_utils.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include "syscall_trace_attach_impl.hpp"
#include <catch2/catch_test_macros.hpp>
#include <boost/scope_exit.hpp>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <climits>
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
		auto provider = bpftime::create_mocked_file_based_on_full_path(
			std::string(root) + "syscalls/sys_enter_read/id");
		REQUIRE(provider.has_value());
		REQUIRE((*provider)->buf == std::to_string(expected) + "\n");
		REQUIRE_FALSE(bpftime::create_mocked_file_based_on_full_path(
			std::string(root) + "syscalls/sys_enter_nonexistent_bpftime/id").has_value());
	}
	FILE *real_id = tmpfile();
	REQUIRE(real_id != nullptr);
	BOOST_SCOPE_EXIT_ALL(=) { fclose(real_id); };
	REQUIRE(fputs("123\n", real_id) >= 0);
	REQUIRE(fflush(real_id) == 0);
	test_tracepoint_fd = fileno(real_id);
	REQUIRE_FALSE(bpftime::create_mocked_file_based_on_full_path(
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
			+[](int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) -> int64_t { return 0; });
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
}
#endif
