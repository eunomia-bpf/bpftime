#include <catch2/catch_test_macros.hpp>
#include <condition_variable>
#include <cstdarg>
#include <new>
#include "../src/seccomp_notifier.cpp"

namespace {
std::atomic<unsigned> receives{ 0 };
std::mutex response_mutex;
std::condition_variable response_ready;
uint32_t response_flags = 0;

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
	response_ready.notify_one();
	return 0;
}

TEST_CASE("Seccomp dispatch failure resumes the intercepted host syscall")
{
	bpftime::seccomp::go_seccomp_listener.store(42);
	bpftime::seccomp::start_go_seccomp_workers();
	std::unique_lock lock(response_mutex);
	REQUIRE(response_ready.wait_for(lock, std::chrono::seconds(3), [] {
		return response_flags == SECCOMP_USER_NOTIF_FLAG_CONTINUE;
	}));
}

extern "C" int __wrap_prctl(int, ...)
{
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
