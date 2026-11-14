#include <catch2/catch_test_macros.hpp>
#include <condition_variable>
#include <cstdarg>
#include <new>
#include <spdlog/sinks/callback_sink.h>
#include "../src/seccomp_notifier.cpp"

namespace {
std::atomic<unsigned> receives{ 0 };
std::mutex response_mutex;
std::condition_variable response_ready;
uint32_t response_flags = 0;
uint64_t response_id = 0;
std::atomic<int> worker_allocations_before_failure{ -1 };
std::atomic<bool> allow_install{ false };

int64_t failing_dispatch(int64_t, int64_t, int64_t, int64_t, int64_t,
			 int64_t, int64_t, int64_t, int64_t, int64_t)
{
	throw std::bad_alloc();
}
}

namespace bpftime {
syscall_hooker_func_t get_call_hook() { return failing_dispatch; }
namespace seccomp {
int64_t raw_syscall(int64_t, int64_t, int64_t, int64_t, int64_t,
		    int64_t, int64_t) { return -ENOSYS; }
}
}

extern "C" int __wrap_ioctl(int, unsigned long operation, ...)
{
	va_list args;
	va_start(args, operation);
	void *data = va_arg(args, void *);
	va_end(args);
	if (operation == SECCOMP_IOCTL_NOTIF_RECV) {
		if (receives.fetch_add(1) == 0) {
			auto *request = static_cast<seccomp_notif *>(data);
			request->id = 123;
			request->data.nr = __NR_bpf;
			return 0;
		}
		errno = EBADF;
		return -1;
	}
	auto *response = static_cast<seccomp_notif_resp *>(data);
	std::lock_guard lock(response_mutex);
	response_flags = response->flags;
	response_id = response->id;
	response_ready.notify_one();
	return 0;
}

TEST_CASE("Seccomp dispatch failure resumes the intercepted host syscall")
{
	bpftime::seccomp::go_seccomp_listener.store(42);
	bpftime::seccomp::start_go_seccomp_workers();
	std::unique_lock lock(response_mutex);
	REQUIRE(response_ready.wait_for(lock, std::chrono::seconds(3), [] {
		return response_flags == SECCOMP_USER_NOTIF_FLAG_CONTINUE &&
		       response_id == 123;
	}));
}

extern "C" int __wrap_prctl(int, ...)
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

// Fail a worker-state allocation after one constructor has already succeeded.
void *operator new(size_t size)
{
	if (worker_allocations_before_failure.load() >= 0 &&
	    worker_allocations_before_failure.fetch_sub(1) == 0) {
		worker_allocations_before_failure = -1;
		throw std::bad_alloc();
	}
	if (void *memory = std::malloc(size ? size : 1))
		return memory;
	throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, size_t) noexcept { std::free(memory); }

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

extern "C" long __wrap_syscall(long number, ...)
{
	if (allow_install.load() && number == SYS_seccomp)
		return 42;
	errno = ENOSYS;
	return -1;
}

TEST_CASE("Post-install logging failure preserves the live seccomp listener")
{
	REQUIRE(setenv("BPFTIME_GO_SECCOMP", "1", 1) == 0);
	auto previous_logger = spdlog::default_logger();
	auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
		[](const spdlog::details::log_msg &) { throw std::bad_alloc(); });
	auto logger = std::make_shared<spdlog::logger>("seccomp-failure-test", sink);
	logger->set_error_handler([](const std::string &) { throw std::bad_alloc(); });
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
