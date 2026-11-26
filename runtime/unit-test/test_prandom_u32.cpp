/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

extern "C" {
uint64_t bpftime_get_prandom_u32();
}

TEST_CASE("bpf_get_prandom_u32 leaves host rand() and errno untouched",
	  "[helpers][prandom]")
{
	srand(12345);
	const int undisturbed_first = rand();
	const int undisturbed_second = rand();

	srand(12345);
	REQUIRE(rand() == undisturbed_first);
	errno = EACCES;
	for (int i = 0; i < 100; i++)
		(void)bpftime_get_prandom_u32();
	REQUIRE(errno == EACCES);
	REQUIRE(rand() == undisturbed_second);
}

TEST_CASE("bpf_get_prandom_u32 draws distinct streams after fork",
	  "[helpers][prandom]")
{
	// Ensure parent TLS PRNG is already seeded before fork().
	(void)bpftime_get_prandom_u32();

	constexpr size_t N = 8;
	int pipefd[2];
	REQUIRE(pipe(pipefd) == 0);

	for (int child = 0; child < 2; child++) {
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0) {
			close(pipefd[0]);
			uint32_t drawn[N];
			for (size_t i = 0; i < N; i++)
				drawn[i] = static_cast<uint32_t>(
					bpftime_get_prandom_u32());
			ssize_t off = 0;
			while (off < static_cast<ssize_t>(sizeof(drawn))) {
				ssize_t n = write(
					pipefd[1],
					reinterpret_cast<const char *>(drawn) +
						off,
					sizeof(drawn) - off);
				if (n < 0)
					_exit(1);
				off += n;
			}
			close(pipefd[1]);
			_exit(0);
		}
	}

	close(pipefd[1]);
	uint32_t seq[2][N];
	for (int child = 0; child < 2; child++) {
		ssize_t off = 0;
		while (off < static_cast<ssize_t>(sizeof(seq[0]))) {
			ssize_t n = read(
				pipefd[0],
				reinterpret_cast<char *>(seq[child]) + off,
				sizeof(seq[0]) - off);
			REQUIRE(n > 0);
			off += n;
		}
		int status = 0;
		REQUIRE(waitpid(-1, &status, 0) > 0);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
	}
	close(pipefd[0]);

	REQUIRE(memcmp(seq[0], seq[1], sizeof(seq[0])) != 0);
}
