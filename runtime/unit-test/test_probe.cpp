#include "bpftime_helper_group.hpp"
#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <signal.h>

static volatile sig_atomic_t forwarded_sigsegv = 0;

static void count_sigsegv(int)
{
	forwarded_sigsegv = forwarded_sigsegv + 1;
}

static void count_replaced_sigsegv(int)
{
	forwarded_sigsegv = forwarded_sigsegv + 2;
}

extern "C" {

int64_t bpftime_probe_read(uint64_t dst, int64_t size, uint64_t ptr, uint64_t,
			   uint64_t);
int64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, int64_t len,
				 uint64_t, uint64_t);
int64_t bpf_probe_read_str(uint64_t dst, uint64_t size, uint64_t src, uint64_t,
			   uint64_t);
uint64_t bpftime_get_current_task(uint64_t, uint64_t, uint64_t, uint64_t,
				  uint64_t);
}

TEST_CASE("Test current task helper exposes the thread pointer")
{
	auto helper_ids = bpftime::bpftime_helper_group::
			  get_kernel_utils_helper_group()
			  .get_helper_ids();
	REQUIRE(std::find(helper_ids.begin(), helper_ids.end(), 35) !=
		helper_ids.end());
	REQUIRE(std::find(helper_ids.begin(), helper_ids.end(), 127) !=
		helper_ids.end());
	REQUIRE(std::find(helper_ids.begin(), helper_ids.end(), 158) !=
		helper_ids.end());
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
	constexpr size_t tpbase_offset = 128;
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET", "128", 1) == 0);
	auto *task = reinterpret_cast<const uint8_t *>(
		bpftime_get_current_task(0, 0, 0, 0, 0));
	void *observed = nullptr;
	memcpy(&observed, task + tpbase_offset, sizeof(observed));
	REQUIRE(observed == __builtin_thread_pointer());
	REQUIRE(unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET") == 0);
#endif
}

TEST_CASE("Test bpftime_probe_read") // test for bpftime_probe_read
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret =
		bpftime_probe_read((uint64_t)dst, size, (uint64_t)src, 0, 0);
	REQUIRE(ret == 0);
	size_t len = sizeof(src) / sizeof(src[0]);
	for (size_t i = 0; i < len; i++) {
		REQUIRE(dst[i] == src[i]);
	}
#ifdef ENABLE_PROBE_READ_CHECK
	ret = bpftime_probe_read((uint64_t)dst, size, (uint64_t)(nullptr), 0,
				 0);
	REQUIRE(ret == -EFAULT);

	ret = bpftime_probe_read((uint64_t)(nullptr), size, (uint64_t)(nullptr),
				 0, 0);
	REQUIRE(ret == -EFAULT);
#endif
}

TEST_CASE("Test repeated valid probe memory access")
{
	int src[4] = { 1, 2, 3, 4 };
	int read_dst[4] = { 0 };
	int write_dst[4] = { 0 };
	uint64_t size = sizeof(src);

	for (size_t i = 0; i < 4; i++) {
		std::memset(read_dst, 0, sizeof(read_dst));
		std::memset(write_dst, 0, sizeof(write_dst));
		REQUIRE(bpftime_probe_read((uint64_t)read_dst, size,
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(std::memcmp(read_dst, src, size) == 0);
		REQUIRE(bpftime_probe_write_user((uint64_t)write_dst,
						 (uint64_t)src, size, 0,
						 0) == 0);
		REQUIRE(std::memcmp(write_dst, src, size) == 0);
	}
}

TEST_CASE("Test bpf_probe_read_str")
{
	char destination[8] = {};
	const char source[] = "text";

	REQUIRE(bpf_probe_read_str((uint64_t)destination, sizeof(destination),
				   (uint64_t)source, 0, 0) == sizeof(source));
	REQUIRE(std::strcmp(destination, source) == 0);

	std::memset(destination, 'x', sizeof(destination));
	REQUIRE(bpf_probe_read_str((uint64_t)destination, 3, (uint64_t)source,
				   0, 0) == 3);
	REQUIRE(std::strcmp(destination, "te") == 0);
	REQUIRE(bpf_probe_read_str((uint64_t)destination, 0, (uint64_t)source,
				   0, 0) == 0);
#ifdef ENABLE_PROBE_READ_CHECK
	std::memset(destination, 'x', sizeof(destination));
	REQUIRE(bpf_probe_read_str((uint64_t)destination, sizeof(destination),
				   0, 0, 0) == -EFAULT);
	REQUIRE(destination[0] == '\0');
	REQUIRE(bpf_probe_read_str(0, sizeof(destination), (uint64_t)source, 0,
				   0) == -EFAULT);

	long page_size = sysconf(_SC_PAGESIZE);
	REQUIRE(page_size > 0);
	auto pages = (char *)mmap(nullptr, (size_t)page_size * 2,
				  PROT_READ | PROT_WRITE,
				  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	REQUIRE(pages != MAP_FAILED);
	pages[page_size - 2] = 'a';
	pages[page_size - 1] = '\0';
	REQUIRE(mprotect(pages + page_size, (size_t)page_size, PROT_NONE) == 0);
	REQUIRE(bpf_probe_read_str((uint64_t)destination, sizeof(destination),
				   (uint64_t)(pages + page_size - 2), 0,
				   0) == 2);
	REQUIRE(std::strcmp(destination, "a") == 0);

	pages[page_size - 1] = 'b';
	std::memset(destination, 'x', sizeof(destination));
	REQUIRE(bpf_probe_read_str((uint64_t)destination, sizeof(destination),
				   (uint64_t)(pages + page_size - 2), 0,
				   0) == -EFAULT);
	for (char value : destination)
		REQUIRE(value == '\0');
	REQUIRE(munmap(pages, (size_t)page_size * 2) == 0);
#endif
}

TEST_CASE("Test probe access preserves application SIGSEGV handlers")
{
	int parent_src = 1;
	int parent_dst = 0;
	REQUIRE(bpftime_probe_read((uint64_t)&parent_dst, sizeof(parent_dst),
				   (uint64_t)&parent_src, 0, 0) == 0);

	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		struct rlimit no_core = { 0, 0 };
		struct sigaction application_handler = {};
		setrlimit(RLIMIT_CORE, &no_core);
		application_handler.sa_handler = count_sigsegv;
		sigemptyset(&application_handler.sa_mask);
		if (sigaction(SIGSEGV, &application_handler, nullptr) != 0)
			_exit(2);

		int src = 1;
		int dst = 0;
		if (bpftime_probe_read((uint64_t)&dst, sizeof(dst),
				       (uint64_t)&src, 0, 0) != 0 ||
		    dst != src)
			_exit(3);
		int raise_result = -1;
		std::thread worker(
			[&raise_result]() { raise_result = raise(SIGSEGV); });
		worker.join();
		if (raise_result != 0 || forwarded_sigsegv != 1)
			_exit(4);

		application_handler.sa_handler = count_replaced_sigsegv;
		if (sigaction(SIGSEGV, &application_handler, nullptr) != 0)
			_exit(5);
		if (bpftime_probe_read((uint64_t)&dst, sizeof(dst),
				       (uint64_t)&src, 0, 0) != 0)
			_exit(6);

		raise_result = raise(SIGSEGV);
		_exit(raise_result == 0 && forwarded_sigsegv == 3 ? 0 : 7);
	}

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("Test bpftime_probe_write_user") // test for bpftime_probe_write_user
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)src,
					       size, 0, 0);
	REQUIRE(ret == 0);
	size_t len = 4;
	for (size_t i = 0; i < len; i++) {
		REQUIRE(dst[i] == src[i]);
	}

#ifdef ENABLE_PROBE_WRITE_CHECK

	ret = bpftime_probe_write_user((uint64_t)(nullptr), (uint64_t)(src),
				       size, 0, 0);
	REQUIRE(ret == -EFAULT);

	ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)(nullptr), size,
				       0, 0);
	REQUIRE(ret == -EFAULT);

#endif

	void *dst1 = (void *)(dst);
	void *src1 = (void *)(src);
	ret = bpftime_probe_write_user((uint64_t)dst1, (uint64_t)src1, size, 0,
				       0);
	REQUIRE(ret == 0);
	for (size_t i = 0; i < len; i++) {
		REQUIRE(((int *)dst1)[i] == ((int *)src1)[i]);
	}
}

TEST_CASE("Test Probe read/write size valid or not ")
{
	int dst[4] = { 0 };
	int src[4] = { 1, 2, 3, 4 };
	uint64_t size = sizeof(src);
	int64_t ret =
		bpftime_probe_read((uint64_t)dst, -1, (uint64_t)src, 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_write_user((uint64_t)dst, (uint64_t)src, -1, 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_read((uint64_t)dst, -1, (uint64_t)(nullptr), 0, 0);
	REQUIRE(ret == -EFAULT);
	ret = bpftime_probe_read((uint64_t)dst, size, (uint64_t)(src), 0, 0);
	REQUIRE(ret == 0);
	for (size_t i = 0; i < 4; i++) {
		REQUIRE(dst[i] == src[i]);
	}
}

TEST_CASE("Test default SIGSEGV handler is preserved")
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		struct sigaction default_action = {};
		default_action.sa_handler = SIG_DFL;
		sigemptyset(&default_action.sa_mask);
		if (sigaction(SIGSEGV, &default_action, nullptr) != 0)
			_exit(2);

		int dst = 0;
		int src = 1;
		if (bpftime_probe_read((uint64_t)&dst, sizeof(dst),
				       (uint64_t)&src, 0, 0) != 0)
			_exit(3);

		struct sigaction current_action = {};
		if (sigaction(SIGSEGV, nullptr, &current_action) != 0)
			_exit(4);
		_exit(current_action.sa_handler == SIG_DFL ? 0 : 5);
	}

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
