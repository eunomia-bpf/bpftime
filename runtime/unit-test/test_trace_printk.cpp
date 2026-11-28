/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>

extern "C" {
uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, uint64_t p1,
			      uint64_t p2, uint64_t p3);
}

// bpf_trace_printk is dispatched by the eBPF VMs as a fixed five-register
// call. On Darwin ARM64 a variadic callee reads its "..." arguments from the
// stack, so a C variadic signature loses p1..p3 (stack garbage). The helper
// must therefore take explicit fixed parameters and print the exact values.
TEST_CASE("bpf_trace_printk forwards explicit register arguments",
	  "[helpers][trace_printk]")
{
	static constexpr const char fmt[] = "p1=%llu p2=%llu p3=%llu\n";
	constexpr uint64_t P1 = 1122334455;
	constexpr uint64_t P2 = 2233445566;
	constexpr uint64_t P3 = 3344556677;
	constexpr const char expected[] =
		"p1=1122334455 p2=2233445566 p3=3344556677\n";

	int saved_stdout = dup(STDOUT_FILENO);
	REQUIRE(saved_stdout >= 0);
	// Flush buffered framework output so it cannot leak into the capture.
	fflush(stdout);
	int pipefd[2];
	REQUIRE(pipe(pipefd) == 0);
	int redirect_rc = dup2(pipefd[1], STDOUT_FILENO);
	close(pipefd[1]);

	// Keep this window free of Catch2 assertions: success reporters
	// (-s/--success) would otherwise write into the captured pipe.
	uint64_t helper_rc = 0;
	if (redirect_rc >= 0) {
		helper_rc = bpftime_trace_printk(
			reinterpret_cast<uint64_t>(fmt), sizeof(fmt), P1, P2,
			P3);
		fflush(stdout);
	}
	int restore_rc = dup2(saved_stdout, STDOUT_FILENO);
	close(saved_stdout);

	char buf[sizeof(expected) + 16] = {};
	size_t off = 0;
	ssize_t n;
	while (off < sizeof(buf) - 1 &&
	       (n = read(pipefd[0], buf + off, sizeof(buf) - 1 - off)) > 0)
		off += static_cast<size_t>(n);
	close(pipefd[0]);

	REQUIRE(redirect_rc >= 0);
	REQUIRE(restore_rc >= 0);
	REQUIRE(helper_rc == 0);
	REQUIRE(off == sizeof(expected) - 1);
	REQUIRE(std::memcmp(buf, expected, off) == 0);
}
