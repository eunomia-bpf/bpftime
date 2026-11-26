#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
#include <bpf/btf.h>
#include <bpf/libbpf.h>
#include <malloc.h>
#endif
#include "bpftime_helper_group.hpp"
#include "base_attach_impl.hpp"
#include <algorithm>
#include "catch2/catch_test_macros.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <atomic>
#include <vector>
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

extern "C" uint64_t bpftime_get_current_task(uint64_t, uint64_t, uint64_t,
	uint64_t, uint64_t);
extern "C" long bpftime_get_current_task_tpbase_offset() noexcept;
extern "C" uint64_t bpftime_seq_write(uint64_t, uint64_t, uint64_t);

TEST_CASE("Task and seq_write helpers are registered")
{
	auto ids = bpftime::bpftime_helper_group::get_kernel_utils_helper_group().get_helper_ids();
	for (int id : { 35, 127, 158 })
		REQUIRE(std::find(ids.begin(), ids.end(), id) != ids.end());
	REQUIRE(static_cast<int64_t>(bpftime_seq_write(0, 0, 0)) == -EOPNOTSUPP);
}

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))

static bool task_test_no_btf = false;
extern "C" struct btf *__real_btf__parse_raw(const char *);
extern "C" struct btf *__wrap_btf__parse_raw(const char *path)
{
	return task_test_no_btf ? nullptr : __real_btf__parse_raw(path);
}

TEST_CASE("Task storage covers host BTF and the no-BTF fallback")
{
	bool no_btf = false;
	SECTION("host BTF") {}
	SECTION("no BTF") { no_btf = true; }
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		size_t expected = 64 * 1024;
		if (!no_btf) {
			struct btf *btf = __real_btf__parse_raw("/sys/kernel/btf/vmlinux");
			if (btf != nullptr && libbpf_get_error(btf) == 0) {
				int id = btf__find_by_name_kind(btf, "task_struct", BTF_KIND_STRUCT);
				int64_t size = id > 0 ? btf__resolve_size(btf, id) : -1;
				if (size > 0)
					expected = size;
				btf__free(btf);
			}
		}
		task_test_no_btf = no_btf;
		const size_t offset = expected - sizeof(void *);
		const std::string text = std::to_string(offset);
		setenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET", text.c_str(), 1);
		auto *task = reinterpret_cast<uint8_t *>(bpftime_get_current_task(0, 0, 0, 0, 0));
		if (task == nullptr || malloc_usable_size(task) < expected ||
		    reinterpret_cast<uintptr_t>(task) % 64 != 0)
			_exit(1);
		void *observed = nullptr;
		memcpy(&observed, task + offset, sizeof(observed));
		if (observed != __builtin_thread_pointer())
			_exit(2);
		// Check late fields before the thread-pointer slot without reading
		// beyond the allocation, including with the old 4 KiB buffer.
		if (!std::all_of(task, task + offset, [](uint8_t b) { return b == 0; }))
			_exit(3);
		if (task != reinterpret_cast<uint8_t *>(bpftime_get_current_task(0, 0, 0, 0, 0)))
			_exit(4);
		bool separate = false;
		std::thread worker([&] {
			auto *other = reinterpret_cast<uint8_t *>(bpftime_get_current_task(0, 0, 0, 0, 0));
			void *tp = nullptr;
			if (other && malloc_usable_size(other) >= expected)
				memcpy(&tp, other + offset, sizeof(tp));
			separate = other != task && tp == __builtin_thread_pointer();
		});
		worker.join();
		_exit(separate ? 0 : 5);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("Task thread-pointer slot comes from host BTF without configuration")
{
	bool no_btf = false;
	SECTION("host BTF") {}
	SECTION("no BTF") { no_btf = true; }
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		task_test_no_btf = no_btf;
		unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
		auto *task = reinterpret_cast<uint8_t *>(bpftime_get_current_task(0, 0, 0, 0, 0));
		if (task == nullptr)
			_exit(1);
		const size_t size = malloc_usable_size(task);
		void *const tp = __builtin_thread_pointer();
		int slots = 0;
		size_t slot_offset = SIZE_MAX;
		for (size_t i = 0; i + sizeof(tp) <= size;
		     i += sizeof(tp)) {
			void *value = nullptr;
			memcpy(&value, task + i, sizeof(value));
			if (value == tp) {
				slots++;
				slot_offset = i;
			}
		}
		// BTF resolves the real task_struct slot; without BTF the
		// fixed fallback slot holds the thread pointer, always at the
		// exported offset.
		_exit(slots == 1 &&
			      slot_offset == (size_t)
					      bpftime_get_current_task_tpbase_offset()
		      ? 0
		      : 2);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

#if defined(__linux__) && defined(ENABLE_PROBE_READ_CHECK)
namespace
{
static void fill_rolling_pattern(uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = static_cast<uint8_t>(i * 31 + 7);
}
} // namespace

TEST_CASE("Test probe read-ahead serves overlapping reads in one event")
{
	const long page_size = sysconf(_SC_PAGESIZE);
	REQUIRE(page_size > 0);
	auto *page = static_cast<uint8_t *>(
		mmap(nullptr, (size_t)page_size, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	REQUIRE(page != MAP_FAILED);
	fill_rolling_pattern(page, (size_t)page_size);

	uint8_t dst[32] = {};
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(scope.entered());
		// First read populates the widened window.
		REQUIRE(bpftime_probe_read((uint64_t)dst, 16, (uint64_t)page, 0,
					   0) == 0);
		REQUIRE(std::memcmp(dst, page, 16) == 0);
		// Overlapping read is served from the captured window.
		std::memset(dst, 0, sizeof(dst));
		REQUIRE(bpftime_probe_read((uint64_t)dst, 16,
					   (uint64_t)(page + 8), 0, 0) == 0);
		REQUIRE(std::memcmp(dst, page + 8, 16) == 0);
		// Adjacent read inside the same page also hits the window.
		std::memset(dst, 0, sizeof(dst));
		REQUIRE(bpftime_probe_read((uint64_t)dst, 8,
					   (uint64_t)(page + 24), 0, 0) == 0);
		REQUIRE(std::memcmp(dst, page + 24, 8) == 0);
	}
	// Outside an event the helper keeps its direct path and fresh data.
	page[0] ^= 0x5A;
	std::memset(dst, 0, sizeof(dst));
	REQUIRE(bpftime_probe_read((uint64_t)dst, 8, (uint64_t)page, 0, 0) == 0);
	REQUIRE(dst[0] == page[0]);
	REQUIRE(munmap(page, (size_t)page_size) == 0);
}

TEST_CASE("Test probe read-ahead keeps one coherent view per event")
{
	alignas(32) uint64_t src[4] = { 1, 2, 3, 4 };
	uint64_t dst[4] = {};
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(scope.entered());
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 1);
		// A same-thread store during the event cannot change bytes the
		// window already captured: the event observes one coherent view.
		src[0] = 99;
		std::memset(dst, 0, sizeof(dst));
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 1);
	}
	// A new event re-reads the current memory.
	std::memset(dst, 0, sizeof(dst));
	REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src), (uint64_t)src, 0,
				   0) == 0);
	REQUIRE(dst[0] == 99);
}

TEST_CASE("Test probe read-ahead is fresh across separate events")
{
	alignas(32) uint64_t src[4] = { 1, 2, 3, 4 };
	uint64_t dst[4] = {};
	// First event populates the window for this address.
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(scope.entered());
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 1);
	}
	// The target changes after the event finished.
	src[0] = 77;
	// A new event at the same overlapping address must observe the new
	// bytes, not the previous event's cached snapshot.
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(scope.entered());
		std::memset(dst, 0, sizeof(dst));
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 77);
	}
}

TEST_CASE("Test probe read-ahead tolerates an inaccessible next page")
{
	const long page_size = sysconf(_SC_PAGESIZE);
	REQUIRE(page_size > 0);
	auto *pages = static_cast<uint8_t *>(
		mmap(nullptr, (size_t)page_size * 2, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	REQUIRE(pages != MAP_FAILED);
	fill_rolling_pattern(pages, (size_t)page_size);
	REQUIRE(mprotect(pages + page_size, (size_t)page_size, PROT_NONE) == 0);

	uint8_t dst[32] = {};
	{
		bpftime::attach::attach_callback_scope scope;
		// A legal read fully inside the readable page, adjacent to the
		// PROT_NONE boundary, must still succeed.
		uint8_t *near = pages + page_size - 16;
		REQUIRE(bpftime_probe_read((uint64_t)dst, 16, (uint64_t)near, 0,
					   0) == 0);
		REQUIRE(std::memcmp(dst, near, 16) == 0);
		// Reading the inaccessible page returns EFAULT, not a crash.
		REQUIRE(bpftime_probe_read((uint64_t)dst, 8,
					   (uint64_t)(pages + page_size), 0,
					   0) == -EFAULT);
	}
	REQUIRE(mprotect(pages + page_size, (size_t)page_size,
			 PROT_READ | PROT_WRITE) == 0);
	REQUIRE(munmap(pages, (size_t)page_size * 2) == 0);
}

TEST_CASE("Test probe read-ahead returns EFAULT for a bad source in an event")
{
	uint64_t dst[2] = {};
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(dst),
					   (uint64_t)nullptr, 0, 0) ==
		-EFAULT);
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(dst),
					   (uint64_t)0x1000, 0, 0) == -EFAULT);
	}
	// After a fault, a valid read in a later event still works.
	uint64_t src[2] = { 7, 8 };
	REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src), (uint64_t)src, 0,
				   0) == 0);
	REQUIRE(dst[0] == 7);
	REQUIRE(dst[1] == 8);
}

// Not gated on ENABLE_PROBE_WRITE_CHECK: read-window invalidation must run in
// bpftime_probe_write_user before both the fault-safe and raw-memcpy branches.
TEST_CASE("Test probe write invalidates the read-ahead window")
{
	uint64_t src[2] = { 11, 22 };
	uint64_t read_dst[2] = {};
	uint64_t new_values[2] = { 33, 44 };
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(bpftime_probe_read((uint64_t)read_dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(read_dst[0] == 11);
		// Overwrite the source through the fault-safe write helper.
		REQUIRE(bpftime_probe_write_user((uint64_t)src,
						  (uint64_t)new_values,
						  sizeof(new_values), 0, 0) == 0);
		// The window was invalidated, so the fresh value is observed.
		REQUIRE(bpftime_probe_read((uint64_t)read_dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(read_dst[0] == 33);
		REQUIRE(read_dst[1] == 44);
	}
}

TEST_CASE("Test probe read-ahead falls back for reads larger than the window")
{
	// 256 bytes exceeds the 128-byte window, so the exact path is used and the
	// result must match the source regardless of window state.
	constexpr size_t n = 256;
	auto *src = new uint8_t[n];
	auto *dst = new uint8_t[n]();
	fill_rolling_pattern(src, n);
	{
		bpftime::attach::attach_callback_scope scope;
		// Prime the window with a small read first.
		REQUIRE(bpftime_probe_read((uint64_t)dst, 16, (uint64_t)src, 0,
					   0) == 0);
		std::memset(dst, 0, n);
		REQUIRE(bpftime_probe_read((uint64_t)dst, n, (uint64_t)src, 0,
					   0) == 0);
		REQUIRE(std::memcmp(dst, src, n) == 0);
		// A following small read still behaves correctly.
		std::memset(dst, 0, 16);
		REQUIRE(bpftime_probe_read((uint64_t)dst, 16, (uint64_t)src, 0,
					   0) == 0);
		REQUIRE(std::memcmp(dst, src, 16) == 0);
	}
	delete[] src;
	delete[] dst;
}

TEST_CASE("Test probe read-ahead keeps event coherence for direct stores")
{
	// A store that bypasses bpf_probe_write_user (e.g. a direct memory write)
	// is not observable until the next event: within an event the helper
	// presents one coherent snapshot.
	uint64_t src[2] = { 5, 6 };
	uint64_t dst[2] = {};
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 5);
		src[0] = 42; // direct mutation, not through probe_write_user
		std::memset(dst, 0, sizeof(dst));
		REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src),
					   (uint64_t)src, 0, 0) == 0);
		REQUIRE(dst[0] == 5); // snapshot within the same event
	}
	std::memset(dst, 0, sizeof(dst));
	REQUIRE(bpftime_probe_read((uint64_t)dst, sizeof(src), (uint64_t)src, 0,
				   0) == 0);
	REQUIRE(dst[0] == 42); // fresh fetch in the next event
}

TEST_CASE("Test probe read-ahead stays independent across threads")
{
	constexpr int kThreads = 8;
	constexpr int kIters = 2000;
	std::vector<std::thread> pool;
	std::atomic<int> failures{ 0 };
	for (int t = 0; t < kThreads; t++) {
		pool.emplace_back([t, &failures] {
			uint64_t src[64];
			for (uint64_t i = 0; i < 64; i++)
				src[i] = (uint64_t)t * 1000 + i;
			uint64_t dst[64] = {};
			for (int it = 0; it < kIters; it++) {
				{
					bpftime::attach::attach_callback_scope
						scope;
					for (size_t off = 0; off + 2 <= 64;
					     off += 2) {
						if (bpftime_probe_read(
							    (uint64_t)dst, 16,
							    (uint64_t)(src +
								       off),
							    0, 0) != 0 ||
						    dst[0] != src[off] ||
						    dst[1] != src[off + 1]) {
							failures.fetch_add(1);
							return;
						}
					}
				}
			}
		});
	}
	for (auto &th : pool)
		th.join();
	REQUIRE(failures.load() == 0);
}

TEST_CASE("Test probe read keeps fault-safe destination outside an event")
{
	// Without an attach scope the exact process_vm_readv path is used, so an
	// invalid destination returns -EFAULT instead of crashing the host.
	uint64_t src[2] = { 1, 2 };
	REQUIRE(bpftime_probe_read((uint64_t) nullptr, sizeof(src),
				   (uint64_t)src, 0, 0) == -EFAULT);
}

TEST_CASE("Test probe read-ahead pins the in-event destination trust contract")
{
	// Inside an event a small read is served through the window and landed by
	// memcpy. The destination is the BPF program's own trusted buffer; an
	// intentionally invalid in-event destination is a contract violation that
	// raises SIGSEGV in the child, pinned here via fork so the test host lives.
	alignas(16) uint64_t src[2] = { 1, 2 };
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		signal(SIGSEGV, SIG_DFL);
		bpftime::attach::attach_callback_scope scope;
		if (!scope.entered())
			_exit(20);
		// Never returns: valid source, invalid destination through memcpy.
		(void)bpftime_probe_read((uint64_t) nullptr, sizeof(src),
					 (uint64_t)src, 0, 0);
		_exit(21);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGSEGV);
}

TEST_CASE("Test probe read-ahead returns EFAULT for a cross-page straddle")
{
	// A request that starts in a readable page but straddles into a PROT_NONE
	// page cannot use the page-clamped window and must fall back to the exact
	// syscall path, which reports -EFAULT for the unreadable tail.
	const long page_size = sysconf(_SC_PAGESIZE);
	REQUIRE(page_size > 0);
	auto *pages = static_cast<uint8_t *>(
		mmap(nullptr, (size_t)page_size * 2, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	REQUIRE(pages != MAP_FAILED);
	fill_rolling_pattern(pages, (size_t)page_size);
	REQUIRE(mprotect(pages + page_size, (size_t)page_size, PROT_NONE) == 0);

	auto *dst = new uint64_t[8]();
	// Place a 32-byte request 16 bytes before the protected boundary.
	void *straddle = pages + page_size - 16;
	{
		bpftime::attach::attach_callback_scope scope;
		REQUIRE(scope.entered());
		REQUIRE(bpftime_probe_read((uint64_t)dst, 32, (uint64_t)straddle, 0,
					   0) == -EFAULT);
	}
	delete[] dst;
	REQUIRE(mprotect(pages + page_size, (size_t)page_size,
			 PROT_READ | PROT_WRITE) == 0);
	REQUIRE(munmap(pages, (size_t)page_size * 2) == 0);
}
#endif
#endif
