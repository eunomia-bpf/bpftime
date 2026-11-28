// Rootless seccomp USER_NOTIF failure-path tests, ported from the deleted
// attach/seccomp_attach_impl/test/test_seccomp_failure.cpp after the notifier
// moved into the self-contained runtime/syscall-server.
//
// Adaptation: the cross-component get_call_hook() seam was removed and the
// object-local dispatch_syscall() is noexcept, so a dispatch failure now
// reaches the worker catch(...) through a throwing allocation in the
// non-noexcept mapped-file resolver instead of a throwing hook function.
// The safety contract under test is unchanged: a failed notification is
// answered with SECCOMP_USER_NOTIF_FLAG_CONTINUE so the host thread is never
// abandoned, and failed/partial filter install publishes listener state -2.

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <new>
#include <spdlog/sinks/callback_sink.h>
#include <string>
#include <vector>

#include "../seccomp_notifier.cpp"

namespace {
std::mutex response_mutex;
std::condition_variable response_ready;
uint32_t response_flags = 0;
uint64_t response_id = 0;
std::atomic<bool> allow_install{ false };
// Armed on the worker thread inside its RECV wrapper.
thread_local bool fail_worker_next_alloc = false;
std::atomic<int> worker_allocations_before_failure{ -1 };
std::atomic<int> seam_tgkill_rc{ 0 };
std::atomic<bool> seam_reject_continue{ false };
std::atomic<bool> seam_close_request{ false };
int emulated_count = 0;

// Long enough to defeat std::string SSO, forcing a heap allocation in
// rootless::resolve_map_files_path() when building the /proc/<pid>/maps path.
const char *mapped_openat_path = "/proc/999999/map_files/7f0000-7f1000";

// Copy of the classic-BPF program handed to the seccomp() syscall, captured
// in __wrap_syscall so test cases can assert on the installed filter itself.
std::vector<sock_filter> captured_filter;
bool filter_captured = false;
} // namespace

long
dispatch_syscall(long, long, long, long, long, long, long) noexcept
{
	return 0;
}

namespace bpftime {
namespace seccomp {
int64_t
raw_syscall(int64_t nr, int64_t, int64_t, int64_t, int64_t, int64_t,
	    int64_t)
{
	return nr == SYS_tgkill ? seam_tgkill_rc.load() : -ENOSYS;
}
} // namespace seccomp
// Standalone seam: the notifier is compiled without syscall_server_utils.cpp
// in this harness, and the failure paths under test never take a redirect.
std::optional<std::string>
resolve_redirected_path(const std::filesystem::path &)
{
	return std::nullopt;
}
} // namespace bpftime

extern "C" bool bpftime_global_shm_is_initialized() { return true; }
extern "C" int bpftime_is_ringbuf_map(int) { return 0; }
extern "C" int bpftime_is_array_map(int) { return 0; }
extern "C" int bpftime_is_software_perf_event(int) { return 0; }
extern "C" int bpftime_is_exist_fake_fd(int) { return 0; }

static bool
alloc_should_throw()
{
	if (fail_worker_next_alloc)
		return true;
	if (worker_allocations_before_failure.load() >= 0 &&
	    worker_allocations_before_failure.fetch_sub(1) == 0) {
		worker_allocations_before_failure = -1;
		return true;
	}
	return false;
}

void *
operator new(size_t size)
{
	if (alloc_should_throw())
		throw std::bad_alloc();
	if (void *memory = std::malloc(size ? size : 1))
		return memory;
	throw std::bad_alloc();
}
void *
operator new[](size_t size)
{
	if (alloc_should_throw())
		throw std::bad_alloc();
	if (void *memory = std::malloc(size ? size : 1))
		return memory;
	throw std::bad_alloc();
}
void
operator delete(void *memory) noexcept
{
	std::free(memory);
}
void
operator delete(void *memory, size_t) noexcept
{
	std::free(memory);
}
void
operator delete[](void *memory) noexcept
{
	std::free(memory);
}

extern "C" int
__wrap_ioctl(int fd, unsigned long operation, ...)
{
	va_list args;
	va_start(args, operation);
	void *data = va_arg(args, void *);
	va_end(args);
	if (operation == SECCOMP_IOCTL_NOTIF_RECV) {
		static thread_local int receives = 0;
		if (receives++ == 0) {
			auto *request = static_cast<seccomp_notif *>(data);
			if (fd == 43 && seam_close_request.load()) {
				request->id = 456;
				request->data.nr = __NR_close;
				request->data.args[0] = 3;
				return 0;
			}
			if (fd != 42) { errno = EBADF; return -1; }
			request->id = 123;
			request->data.nr = __NR_openat;
			request->data.args[1] =
				reinterpret_cast<uint64_t>(mapped_openat_path);
			fail_worker_next_alloc = true;
			return 0;
		}
		errno = EBADF;
		return -1;
	}
	auto *response = static_cast<seccomp_notif_resp *>(data);
	if (seam_reject_continue.load() &&
	    (response->flags & SECCOMP_USER_NOTIF_FLAG_CONTINUE) != 0) {
		errno = EINVAL;
		return -1;
	}
	std::lock_guard lock(response_mutex);
	response_flags = response->flags;
	response_id = response->id;
	if (response->id == 456 && response->flags == 0 && response->error == 0)
		emulated_count++;
	fail_worker_next_alloc = false;
	response_ready.notify_one();
	return 0;
}

TEST_CASE("Seccomp emulation failure resumes the intercepted host syscall")
{
	// Self-check: the global replacement must actually intercept allocations.
	fail_worker_next_alloc = true;
	bool self_threw = false;
	try {
		auto *volatile p = new int;
		(void)p;
	} catch (const std::bad_alloc &) {
		self_threw = true;
	}
	fail_worker_next_alloc = false;
	REQUIRE(self_threw);

	bpftime::seccomp::go_seccomp_listener.store(42);
	bpftime::seccomp::start_go_seccomp_workers();
	std::unique_lock lock(response_mutex);
	REQUIRE(response_ready.wait_for(lock, std::chrono::seconds(3), [] {
		return response_flags == SECCOMP_USER_NOTIF_FLAG_CONTINUE &&
		       response_id == 123;
	}));
}

TEST_CASE("Kernel < 5.5 EINVAL on FLAG_CONTINUE falls back to emulation")
{
	{
		std::lock_guard lock(response_mutex);
		emulated_count = 0;
	}
	seam_reject_continue = true;
	seam_close_request = true;
	bpftime::seccomp::flag_continue_supported.store(true);
	bpftime::seccomp::go_seccomp_listener.store(43);
	bpftime::seccomp::start_go_seccomp_workers();
	std::unique_lock lock(response_mutex);
	REQUIRE(response_ready.wait_for(lock, std::chrono::seconds(3), [] {
		return emulated_count == 8;
	}));
	CHECK(!bpftime::seccomp::flag_continue_supported.load());
	seam_close_request = false;
	seam_reject_continue = false;
	bpftime::seccomp::flag_continue_supported.store(true);
}

extern "C" int
__wrap_prctl(int, ...)
{
	if (allow_install.load())
		return 0;
	errno = EACCES;
	return -1;
}

TEST_CASE("Failed seccomp installation cancels waiting workers")
{
	REQUIRE(setenv("BPFTIME_GO_SECCOMP", "1", 1) == 0);
	bpftime::seccomp::go_seccomp_listener.store(-1);
	const bool installed = bpftime::seccomp::install_seccomp_tracer();
	unsetenv("BPFTIME_GO_SECCOMP");
	REQUIRE_FALSE(installed);
	REQUIRE(bpftime::seccomp::go_seccomp_listener.load() == -2);
}

TEST_CASE("Worker construction failure cancels partially started seccomp setup")
{
	REQUIRE(setenv("BPFTIME_GO_SECCOMP", "1", 1) == 0);
	bpftime::seccomp::go_seccomp_listener.store(-1);
	worker_allocations_before_failure = 1;
	const bool installed = bpftime::seccomp::install_seccomp_tracer();
	worker_allocations_before_failure = -1;
	unsetenv("BPFTIME_GO_SECCOMP");
	REQUIRE_FALSE(installed);
	REQUIRE(bpftime::seccomp::go_seccomp_listener.load() == -2);
}

extern "C" long
__wrap_syscall(long number, ...)
{
	va_list args;
	va_start(args, number);
	int operation = va_arg(args, int);
	(void)va_arg(args, unsigned long); // seccomp flags
	void *filter_arg = va_arg(args, void *);
	va_end(args);
	if (allow_install.load() && number == SYS_seccomp) {
		if (operation == SECCOMP_SET_MODE_FILTER &&
		    filter_arg != nullptr) {
			const auto *program =
				static_cast<const sock_fprog *>(filter_arg);
			captured_filter.assign(
				program->filter,
				program->filter + program->len);
			filter_captured = true;
		}
		return 42;
	}
	errno = ENOSYS;
	return -1;
}

TEST_CASE("Post-install logging failure preserves the live seccomp listener")
{
	REQUIRE(setenv("BPFTIME_GO_SECCOMP", "1", 1) == 0);
	auto previous_logger = spdlog::default_logger();
	auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
		[](const spdlog::details::log_msg &) { throw std::bad_alloc(); });
	auto logger = std::make_shared<spdlog::logger>("seccomp-failure-test",
						       sink);
	logger->set_error_handler([](const std::string &) {
		throw std::bad_alloc();
	});
	spdlog::set_default_logger(logger);
	bpftime::seccomp::go_seccomp_listener.store(-1);
	allow_install = true;
	const bool installed = bpftime::seccomp::install_seccomp_tracer();
	allow_install = false;
	spdlog::set_default_logger(previous_logger);
	unsetenv("BPFTIME_GO_SECCOMP");
	REQUIRE(installed);
	REQUIRE(bpftime::seccomp::go_seccomp_listener.load() == 42);
}

TEST_CASE("Foreign-ABI syscalls fail open instead of killing the process")
{
	REQUIRE(setenv("BPFTIME_GO_SECCOMP", "1", 1) == 0);
	bpftime::seccomp::go_seccomp_listener.store(-1);
	captured_filter.clear();
	filter_captured = false;
	allow_install = true;
	const bool installed = bpftime::seccomp::install_seccomp_tracer();
	allow_install = false;
	unsetenv("BPFTIME_GO_SECCOMP");
	REQUIRE(installed);
	REQUIRE(filter_captured);
	REQUIRE(captured_filter.size() >= 4);

	// The program loads seccomp_data.arch and skips the mismatch rule only
	// when the process's native ABI matches.
	REQUIRE(captured_filter[0].code == (BPF_LD | BPF_W | BPF_ABS));
	REQUIRE(captured_filter[1].code == (BPF_JMP | BPF_JEQ | BPF_K));
	REQUIRE(captured_filter[1].k == BPFTIME_SECCOMP_AUDIT_ARCH);
	REQUIRE(captured_filter[1].jt == 1);

	// Regression: a foreign-ABI syscall (e.g. a 32-bit compat syscall) must
	// be handed to the kernel instead of terminating the process with SIGSYS.
	REQUIRE(captured_filter[2].code == (BPF_RET | BPF_K));
	REQUIRE(captured_filter[2].k == SECCOMP_RET_ALLOW);

	// The trailing default rule still allows every unlisted native syscall.
	REQUIRE(captured_filter.back().code == (BPF_RET | BPF_K));
	REQUIRE(captured_filter.back().k == SECCOMP_RET_ALLOW);

	// No installed rule may kill the process.
	for (const struct sock_filter &insn : captured_filter)
		REQUIRE(insn.k != SECCOMP_RET_KILL_PROCESS);
}
