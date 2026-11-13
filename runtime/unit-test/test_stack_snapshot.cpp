// SPDX-License-Identifier: MIT
// Tests for the bounded fault-safe stack snapshot backend.
//
// These tests pin the contract that matters for rootless profiling:
//   * live stack reads inside a capture are served from the snapshot,
//   * stale/out-of-window stack pointers deterministically return -EFAULT
//     instead of dereferencing them (the boost::fiber SIGSEGV scenario),
//   * non-stack mappings keep the compiled-in probe-read semantics,
//   * nested captures restore the outer context and threads are isolated.
#include "catch2/catch_test_macros.hpp"
#include "stack_snapshot.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__) && BPFTIME_HAVE_STACK_SNAPSHOT

extern "C" int64_t bpftime_probe_read(uint64_t dst, int64_t size,
				      uint64_t ptr, uint64_t, uint64_t);

namespace
{

struct stack_mapping {
	uintptr_t begin = 0;
	uintptr_t end = 0;
};

// Resolve the readable mapping containing `address` from /proc/self/maps.
stack_mapping find_mapping(const void *address)
{
	stack_mapping result;
	FILE *maps = fopen("/proc/self/maps", "r");
	REQUIRE(maps != nullptr);
	const uintptr_t target = reinterpret_cast<uintptr_t>(address);
	char line[512];
	while (fgets(line, sizeof(line), maps) != nullptr) {
		uintptr_t low = 0;
		uintptr_t high = 0;
		char perms[5] = {};
		if (sscanf(line, "%lx-%lx %4s", &low, &high, perms) != 3)
			continue;
		if (perms[0] == 'r' && target >= low && target < high) {
			result = { low, high };
			break;
		}
	}
	fclose(maps);
	return result;
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

TEST_CASE("Live stack bytes are served from the bounded snapshot")
{
	// The capture window starts at the hardware SP sampled in the capture's
	// constructor, so `src` must live at or above that SP. With an elided
	// frame the compiler is free to place it at exactly the SP, which the
	// window still covers, but nothing guarantees it lands *after* the
	// constructor's own frame — so the buffer is declared volatile to keep it
	// in the frame, and the capture is given the buffer's own address as its
	// floor by sampling SP from a helper called after `src` exists.
	volatile char src[64];
	char dst[64] = {};

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	// Sample from the buffer's own address so the window is guaranteed to
	// cover it, independent of how the compiler lays out this frame. The
	// runtime path samples the hardware SP, which an elided frame can put
	// exactly at (or, for a sibling local, below) the buffer.
	bpftime::stack_snapshot::stack_sample_capture capture(
		reinterpret_cast<uintptr_t>(const_cast<char *>(src)));
	REQUIRE(capture.active());

	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(src),
				   reinterpret_cast<uint64_t>(const_cast<char *>(src)),
				   0, 0) == 0);
	REQUIRE(std::memcmp(dst, const_cast<char *>(src), sizeof(src)) == 0);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Stale stack pointer below sampled SP returns EFAULT, no dereference")
{
	char anchor[16] = {};
	(void)anchor;
	const stack_mapping mapping = find_mapping(anchor);
	REQUIRE(mapping.begin != 0);

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture;
	REQUIRE(capture.active());
	const auto *sample = bpftime::stack_snapshot::current_sample;
	REQUIRE(sample != nullptr);
	// The first byte of the mapping is mapped and readable, but it lies
	// below the sampled SP: the D-gate must reject it with EFAULT rather
	// than touching it through the per-read fault-safe backend.
	REQUIRE(mapping.begin < sample->sampled_sp);

	char dst[16] = {};
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst),
				   sizeof(dst), mapping.begin, 0, 0) ==
		-EFAULT);
	unsetenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT");
}

TEST_CASE("Reads crossing the captured window end return EFAULT")
{
	char anchor[16] = {};
	(void)anchor;

	setenv("BPFTIME_EXPERIMENT_STACK_SNAPSHOT", "1", 1);
	bpftime::stack_snapshot::stack_sample_capture capture;
	REQUIRE(capture.active());
	const auto *sample = bpftime::stack_snapshot::current_sample;
	REQUIRE(sample != nullptr);
	REQUIRE(sample->window.size() > sizeof(uint64_t));

	const uintptr_t last_four =
		sample->sampled_sp + sample->window.size() - 4;
	uint64_t dst = 0;
	// 8-byte read starting 4 bytes before the window end must be rejected
	// by integer bounds checking, never split across the boundary.
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(&dst),
				   sizeof(dst), last_four, 0, 0) == -EFAULT);
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

	// An unmapped address must be absorbed as EFAULT while the process
	// keeps running: the fiber crash contract.
	REQUIRE(bpftime_probe_read(reinterpret_cast<uint64_t>(dst), 8,
				   0x1000, 0, 0) == -EFAULT);

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
