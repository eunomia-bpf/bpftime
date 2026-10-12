/* SPDX-License-Identifier: MIT */
#include "bpf_helper_read_and_write.hpp"
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sched.h>

static constexpr uint64_t iterations = 100'000'000;
static constexpr uint64_t warmup_iterations = 10'000;
static constexpr size_t copy_size = sizeof(uint64_t);

template <typename F>
static void measure_probe_access(const char *operation, F copy)
{
	cpu_set_t allowed;
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	int cpu = 0;
	while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed))
		++cpu;
	REQUIRE(cpu < CPU_SETSIZE);
	cpu_set_t pinned;
	CPU_ZERO(&pinned);
	CPU_SET(cpu, &pinned);
	REQUIRE(sched_setaffinity(0, sizeof(pinned), &pinned) == 0);

	// Resolve initialization and check that this backend can operate before
	// entering the timed loop. Only successful, valid-address copies count.
	REQUIRE(copy() == 0);
	uint64_t failures = 0;
	for (uint64_t i = 0; i < warmup_iterations; ++i)
		failures += copy() != 0;
	REQUIRE(failures == 0);

	auto start = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < iterations; ++i)
		failures += copy() != 0;
	auto end = std::chrono::steady_clock::now();
	REQUIRE(failures == 0);

	double total_ms =
		std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "PROBE_BENCH backend=" << PROBE_BENCH_BACKEND
		  << " operation=" << operation << " cpu=" << cpu
		  << " bytes=" << copy_size << " iterations=" << iterations
		  << std::fixed << std::setprecision(6)
		  << " total_ms=" << total_ms << std::scientific
		  << std::setprecision(9)
		  << " average_ms=" << total_ms / iterations << '\n'
		  << std::flush;
}

// Hidden tests: one hundred million calls per operation are intentionally
// opt-in, rather than part of the ordinary unit-test run.
TEST_CASE("Benchmark valid probe_read calls", "[.probe-access-benchmark]")
{
	uint64_t source = 0x123456789abcdef0, destination = 0;
	measure_probe_access("read", [&] {
		return bpftime_probe_read_user((uint64_t)&destination,
					       copy_size, (uint64_t)&source, 0,
					       0);
	});
	REQUIRE(destination == source);
}

TEST_CASE("Benchmark valid probe_write calls", "[.probe-access-benchmark]")
{
	uint64_t source = 0x123456789abcdef0, destination = 0;
	measure_probe_access("write", [&] {
		return bpftime_probe_write_user((uint64_t)&destination,
						(uint64_t)&source, copy_size, 0,
						0);
	});
	REQUIRE(destination == source);
}
