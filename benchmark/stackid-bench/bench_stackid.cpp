// Real-helper benchmark for bpf_get_stackid / bpf_get_stack on bpftime.
//
// It drives the public, ABI-stable bpftime_get_stackid()/bpftime_get_stack()
// helpers from inside a REAL Frida uprobe callback, so every timed call runs
// the full production path:
//   frida GumBacktracer generate_stack() walk  ->
//   stack_walk_data populated on the stack     ->
//   stack-trace map fill_stack_trace() update.
//
// Two phases, each reported for master vs. this head on the same machine:
//   1. single-thread: warmup excluded, precise ns/op of each helper plus the
//      host-heap operator new count inside the timed region;
//   2. multi-thread: 4 threads fire the uprobe concurrently (each firing does
//      one get_stackid and one get_stack), reporting contended ns/op. This is
//      the malloc-arena context the deadlock happened in.
//
// A global operator new/delete override counts host-heap allocations. On
// master the helper allocates std::vector per call; the fix replaces it with
// the stack-resident stack_walk_data (zero allocations).
//
// Built by the `stackid-bench` CMake target (BPFTIME_ENABLE_FRIDA=ON, the
// default). Run the resulting binary; it prints RESULT lines.

#include "bpf_attach_ctx.hpp"
#include "bpftime_shm.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "base_attach_impl.hpp"

#include "linux/bpf.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string_view>
#include <thread>
#include <vector>

using namespace bpftime;

// ---- host heap allocation counting -------------------------------------
static std::atomic<unsigned long> g_new_calls{ 0 };

void *operator new(std::size_t n)
{
	g_new_calls.fetch_add(1, std::memory_order_relaxed);
	if (void *p = std::malloc(n))
		return p;
	throw std::bad_alloc();
}
void *operator new[](std::size_t n)
{
	g_new_calls.fetch_add(1, std::memory_order_relaxed);
	if (void *p = std::malloc(n))
		return p;
	throw std::bad_alloc();
}
void operator delete(void *p) noexcept
{
	std::free(p);
}
void operator delete[](void *p) noexcept
{
	std::free(p);
}
void operator delete(void *p, std::size_t) noexcept
{
	std::free(p);
}
void operator delete[](void *p, std::size_t) noexcept
{
	std::free(p);
}

// ---- real Frida attach context, overriding the weak mock in bpf_helper ---
static attach::frida_attach_impl *g_frida = nullptr;

bpf_attach_ctx &get_global_attach_ctx()
{
	static bpf_attach_ctx ctx;
	static const bool initialized = [] {
		// Fuzzy (frame-pointer) backtracer: fault-tolerant when the walk
		// crosses Frida's interceptor trampoline, unlike the libunwind
		// accurate backtracer on aarch64. The #974 caching change is
		// backtracer-agnostic, so either one exercises the same path.
		auto impl = std::make_unique<attach::frida_attach_impl>(true);
		g_frida = impl.get();
		ctx.register_attach_impl(
			{ attach::ATTACH_UPROBE }, std::move(impl),
			[](const std::string_view &, int &) {
				return std::unique_ptr<attach::attach_private_data>();
			});
		return true;
	}();
	(void)initialized;
	return ctx;
}

extern "C" int64_t bpftime_get_stackid(uint64_t, uint64_t, uint64_t, uint64_t,
				       uint64_t);
extern "C" int64_t bpftime_get_stack(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t);

// ---- deep noinline call chain so Gum walks several real frames ----------
extern "C" __attribute__((noinline)) uint64_t bench_leaf(uint64_t x)
{
	asm("");
	return x + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f11(uint64_t x)
{
	asm("");
	return bench_leaf(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f10(uint64_t x)
{
	asm("");
	return bench_f11(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f9(uint64_t x)
{
	asm("");
	return bench_f10(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f8(uint64_t x)
{
	asm("");
	return bench_f9(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f7(uint64_t x)
{
	asm("");
	return bench_f8(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_f6(uint64_t x)
{
	asm("");
	return bench_f7(x) + 1;
}
extern "C" __attribute__((noinline)) uint64_t bench_root(uint64_t x)
{
	asm("");
	return bench_f6(x) + 1;
}

static constexpr int ST_ITERATIONS = 200000;
static constexpr int MT_THREADS = 4;
static constexpr int MT_PER_THREAD = 50000;
static constexpr int SKIP = 2;

static int g_map_fd = -1;
static std::atomic<unsigned long> g_sink{ 0 };
static std::atomic<unsigned long> g_mt_fires{ 0 };
static std::atomic<bool> g_mt_mode{ false };
static std::atomic<unsigned long> g_frames_seen{ 0 };

static void run_single_thread_phase()
{
	std::array<uint64_t, 127> stack_buf{};

	// Warm up: first call can run one-time cold setup; exclude it.
	int64_t warm = bpftime_get_stackid(0, (uint64_t)g_map_fd,
					   BPF_F_USER_STACK, 0, 0);
	g_sink.fetch_add((unsigned long)(warm < 0 ? 0 : warm),
			 std::memory_order_relaxed);

	// ---- bpf_get_stackid: real Gum walk + map update, timed ----
	const unsigned long new0 = g_new_calls.load(std::memory_order_relaxed);
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < ST_ITERATIONS; ++i) {
		int64_t sid = bpftime_get_stackid(
			0, (uint64_t)g_map_fd, BPF_F_USER_STACK | SKIP, 0, 0);
		g_sink.fetch_add((unsigned long)(sid < 0 ? 0 : sid),
				 std::memory_order_relaxed);
	}
	auto t1 = std::chrono::steady_clock::now();
	const unsigned long new1 = g_new_calls.load(std::memory_order_relaxed);

	// ---- bpf_get_stack: real Gum walk + frame copy, timed ----
	int64_t copied = bpftime_get_stack(
		0, (uint64_t)stack_buf.data(),
		stack_buf.size() * sizeof(uint64_t), BPF_F_USER_STACK, 0);
	if (copied > 0)
		g_frames_seen.store((unsigned long)copied / sizeof(uint64_t),
				    std::memory_order_relaxed);

	const unsigned long new2 = g_new_calls.load(std::memory_order_relaxed);
	auto t2 = std::chrono::steady_clock::now();
	for (int i = 0; i < ST_ITERATIONS; ++i) {
		int64_t n = bpftime_get_stack(
			0, (uint64_t)stack_buf.data(),
			stack_buf.size() * sizeof(uint64_t),
			BPF_F_USER_STACK | SKIP, 0);
		g_sink.fetch_add((unsigned long)(n < 0 ? 0 : n),
				 std::memory_order_relaxed);
	}
	auto t3 = std::chrono::steady_clock::now();
	const unsigned long new3 = g_new_calls.load(std::memory_order_relaxed);

	auto ns = [](auto a, auto b) {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
			.count();
	};
	std::printf("RESULT ST get_stackid: %.2f ns/op  heap_new=%lu (over %d)\n",
		    (double)ns(t0, t1) / ST_ITERATIONS, new1 - new0,
		    ST_ITERATIONS);
	std::printf("RESULT ST get_stack:   %.2f ns/op  heap_new=%lu (over %d)\n",
		    (double)ns(t2, t3) / ST_ITERATIONS, new3 - new2,
		    ST_ITERATIONS);
	std::printf("RESULT frames_per_walk=%lu\n",
		    g_frames_seen.load(std::memory_order_relaxed));
}

static void on_leaf(const pt_regs &)
{
	if (!g_mt_mode.load(std::memory_order_relaxed)) {
		// The first, single-threaded firing runs the detailed phase
		// exactly once.
		static bool done = false;
		if (!done) {
			done = true;
			run_single_thread_phase();
		}
		return;
	}

	// Multi-thread phase: one real helper call of each kind per firing.
	thread_local std::array<uint64_t, 127> tls_buf{};
	int64_t sid = bpftime_get_stackid(0, (uint64_t)g_map_fd,
					  BPF_F_USER_STACK, 0, 0);
	int64_t n = bpftime_get_stack(0, (uint64_t)tls_buf.data(),
				      tls_buf.size() * sizeof(uint64_t),
				      BPF_F_USER_STACK, 0);
	g_sink.fetch_add((unsigned long)(sid < 0 ? 0 : sid) +
				 (unsigned long)(n < 0 ? 0 : n),
			 std::memory_order_relaxed);
	g_mt_fires.fetch_add(1, std::memory_order_relaxed);
}

int main()
{
	const std::string shm_name =
		"bpftime_bench_stackid_" + std::to_string(getpid());
	setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	bpf_map_attr attr{};
	attr.type = static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_STACK_TRACE);
	attr.key_size = sizeof(uint32_t);
	attr.value_size = 127 * sizeof(uint64_t);
	attr.max_ents = 1024;
	g_map_fd = bpftime_maps_create(-1, "bench_stack_trace", attr);
	if (g_map_fd < 0) {
		std::fprintf(stderr, "map creation failed: %d\n", g_map_fd);
		return 1;
	}

	(void)get_global_attach_ctx();
	if (g_frida->create_uprobe_at((void *)&bench_leaf, on_leaf) < 0) {
		std::fprintf(stderr, "create_uprobe_at failed\n");
		return 1;
	}

	// Phase 1: one firing on the main thread runs the detailed timing.
	volatile uint64_t r = bench_root(41);
	(void)r;

	// Phase 2: concurrent firings from several threads.
	g_mt_mode.store(true, std::memory_order_relaxed);
	const unsigned long new_before =
		g_new_calls.load(std::memory_order_relaxed);
	auto t0 = std::chrono::steady_clock::now();
	std::vector<std::thread> workers;
	for (int t = 0; t < MT_THREADS; ++t) {
		workers.emplace_back([] {
			volatile uint64_t acc = 0;
			for (int i = 0; i < MT_PER_THREAD; ++i)
				acc += bench_root((uint64_t)i);
			g_sink.fetch_add((unsigned long)acc,
					 std::memory_order_relaxed);
		});
	}
	for (auto &th : workers)
		th.join();
	auto t1 = std::chrono::steady_clock::now();
	const unsigned long new_after =
		g_new_calls.load(std::memory_order_relaxed);

	const unsigned long fires = g_mt_fires.load(std::memory_order_relaxed);
	const double wall_ns =
		(double)std::chrono::duration_cast<std::chrono::nanoseconds>(
			t1 - t0)
			.count();
	std::printf("RESULT MT %d threads %lu fires: %.2f ns/op  heap_new=%lu\n",
		    MT_THREADS, fires,
		    fires ? wall_ns / (double)fires : 0.0,
		    new_after - new_before);
	std::printf("RESULT sink=%lu\n", g_sink.load(std::memory_order_relaxed));

	bpftime_close(g_map_fd);
	bpftime_destroy_global_shm();
	bpftime_remove_global_shm();
	return 0;
}
