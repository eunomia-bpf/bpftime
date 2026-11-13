// SPDX-License-Identifier: MIT
// Tests for the bounded fault-safe stack snapshot backend.
//
// These tests pin the contract that matters for rootless profiling:
//   * reads fully inside the captured window are served from the snapshot,
//   * reads outside the window but inside a readable mapping fall back to the
//     compiled-in backend and still succeed (deep unwind frames must work),
//   * unreadable addresses are absorbed as -EFAULT while the process keeps
//     running (the boost::fiber crash contract),
//   * non-stack mappings keep the compiled-in probe-read semantics,
//   * nested captures restore the outer context and threads are isolated.
//
// The mapping tests pass an explicit target SP into the capture so they do
// not depend on compiler frame layout (debug frames can exceed the 4 KiB
// window, which is exactly the case the fallback must handle).
#include "catch2/catch_test_macros.hpp"
#include "stack_snapshot.hpp"

#include <cerrno>
#include <algorithm>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__linux__) && BPFTIME_HAVE_STACK_SNAPSHOT

extern "C" int64_t bpftime_probe_read(uint64_t dst, int64_t size,
				      uint64_t ptr, uint64_t, uint64_t);

namespace
{

constexpr size_t page_size = 4096;

// mmap `pages` readable/writable anonymous pages and fill them with a
// position-dependent byte pattern.
uint8_t *map_pattern(size_t pages)
{
	void *region =
		mmap(nullptr, pages * page_size, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	REQUIRE(region != MAP_FAILED);
	auto *bytes = static_cast<uint8_t *>(region);
	for (size_t i = 0; i < pages * page_size; i++)
		bytes[i] = static_cast<uint8_t>(i & 0xff);
	return bytes;
}

} // namespace

TEST_CASE("Stack snapshot env gate enables and disables the capture")
{
	char local[64] = {};
	(void)local;

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "0", 1);
	{
		bpftime::stack_snapshot::stack_sample_capture capture;
		REQUIRE_FALSE(capture.active());
		REQUIRE(bpftime::stack_snapshot::current_sample == nullptr);
	}

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	{
		bpftime::stack_snapshot::stack_sample_capture capture;
		REQUIRE(capture.active());
		REQUIRE(bpftime::stack_snapshot::current_sample != nullptr);
	}
	REQUIRE(bpftime::stack_snapshot::current_sample == nullptr);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("In-window reads are served with the captured bytes")
{
	uint8_t *region = map_pattern(3);
	const uintptr_t sp = reinterpret_cast<uintptr_t>(region) + 2048;

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture(sp);
	REQUIRE(capture.active());

	uint8_t dst[64] = {};
	const uintptr_t inside = sp + 64;
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst), inside, 0, 0) == 0);
	for (size_t i = 0; i < sizeof(dst); i++)
		REQUIRE(dst[i] ==
			static_cast<uint8_t>((inside - reinterpret_cast<uintptr_t>(
							      region) +
						      i) &
					     0xff));

	munmap(region, 3 * page_size);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Reads above the captured window fall back and still succeed")
{
	// A deep unwind frame can sit far above the sampled SP. It is inside a
	// readable mapping but outside the 4 KiB window, so the per-read
	// backend must serve it instead of reporting a fault.
	uint8_t *region = map_pattern(3);
	const uintptr_t sp = reinterpret_cast<uintptr_t>(region);

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture(sp);
	REQUIRE(capture.active());

	uint8_t dst[32] = {};
	const uintptr_t deep = sp + 2 * page_size;
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst), deep, 0, 0) == 0);
	for (size_t i = 0; i < sizeof(dst); i++)
		REQUIRE(dst[i] ==
			static_cast<uint8_t>((2 * page_size + i) & 0xff));

	munmap(region, 3 * page_size);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Readable addresses below the sampled SP fall back correctly")
{
	uint8_t *region = map_pattern(3);
	const uintptr_t sp = reinterpret_cast<uintptr_t>(region) + 2048;

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture(sp);
	REQUIRE(capture.active());

	uint8_t dst[16] = {};
	const uintptr_t below = reinterpret_cast<uintptr_t>(region) + 128;
	REQUIRE(below < sp);
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst), below, 0, 0) == 0);
	for (size_t i = 0; i < sizeof(dst); i++)
		REQUIRE(dst[i] == static_cast<uint8_t>((128 + i) & 0xff));

	munmap(region, 3 * page_size);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Unreadable addresses return EFAULT while the process survives")
{
	uint8_t *region = map_pattern(2);
	REQUIRE(mprotect(region + page_size, page_size, PROT_NONE) == 0);
	const uintptr_t sp = reinterpret_cast<uintptr_t>(region);

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture(sp);
	REQUIRE(capture.active());

	uint8_t dst[16] = {};
	// Guarded page: outside the window, fallback pvmr absorbs the fault.
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst),
				   reinterpret_cast<uint64_t>(
					   region + page_size),
				   0, 0) == -EFAULT);
	// Unmapped low address keeps the same guarantee.
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst), 0x1000, 0, 0) == -EFAULT);

	munmap(region, 2 * page_size);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Non-stack mappings fall back to the compiled-in backend")
{
	const size_t size = 128;
	void *heap = std::malloc(size);
	REQUIRE(heap != nullptr);
	std::memset(heap, 0x7a, size);

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture;
	REQUIRE(capture.active());

	char dst[128] = {};
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst), size,
				   reinterpret_cast<uint64_t>(heap), 0, 0) ==
		0);
	REQUIRE(std::all_of(dst, dst + size,
			    [](char c) { return c == static_cast<char>(0x7a); }));

	std::free(heap);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Nested captures restore the outer replay context")
{
	char outer_src[32] = {};
	for (size_t i = 0; i < sizeof(outer_src); i++)
		outer_src[i] = static_cast<char>(i + 7);

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture outer;
	REQUIRE(outer.active());
	const auto *outer_context = bpftime::stack_snapshot::current_sample;
	REQUIRE(outer_context != nullptr);

	{
		bpftime::stack_snapshot::stack_sample_capture inner;
		REQUIRE(inner.active());
		REQUIRE(bpftime::stack_snapshot::current_sample !=
			outer_context);

		char dst[32] = {};
		REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
					   sizeof(outer_src),
					   reinterpret_cast<uint64_t>(outer_src),
					   0, 0) == 0);
		REQUIRE(std::memcmp(dst, outer_src, sizeof(outer_src)) == 0);
	}
	REQUIRE(bpftime::stack_snapshot::current_sample == outer_context);
	char dst[32] = {};
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(outer_src),
				   reinterpret_cast<uint64_t>(outer_src), 0,
				   0) == 0);
	REQUIRE(std::memcmp(dst, outer_src, sizeof(outer_src)) == 0);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Per-thread snapshot contexts are isolated")
{
	constexpr int thread_count = 4;
	std::vector<std::thread> threads;
	std::atomic<int> ok_count{ 0 };

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	for (int t = 0; t < thread_count; t++) {
		threads.emplace_back([&ok_count] {
			char local[64];
			for (size_t i = 0; i < sizeof(local); i++)
				local[i] = static_cast<char>(i + 3);
			bpftime::stack_snapshot::stack_sample_capture capture;
			if (!capture.active())
				return;
			char dst[64] = {};
			if (bpftime_probe_read(
				    reinterpret_cast<uint64_t>(dst),
				    sizeof(local),
				    reinterpret_cast<uint64_t>(local), 0,
				    0) != 0)
				return;
			if (std::memcmp(dst, local, sizeof(local)) == 0)
				ok_count.fetch_add(1);
		});
	}
	for (auto &thread : threads)
		thread.join();
	REQUIRE(ok_count.load() == thread_count);
	REQUIRE(bpftime::stack_snapshot::current_sample == nullptr);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

#else

TEST_CASE("Stack snapshot backend is compiled out on unsupported platforms")
{
	SUCCEED();
}

#endif
