/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#if defined(__linux__)
#include "robust_shm_mutex.hpp"
#include <cerrno>
#include <new>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

TEST_CASE("robust SHM mutex taints permanently after a SIGKILLed holder",
	  "[robust_shm_mutex]")
{
	void *p = mmap(nullptr, sizeof(bpftime::robust_shm_mutex),
		       PROT_READ | PROT_WRITE,
		       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	REQUIRE(p != MAP_FAILED);
	auto *m = new (p) bpftime::robust_shm_mutex();

	int notify[2];
	REQUIRE(pipe(notify) == 0);
	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(notify[0]);
		if (!m->lock("test"))
			_exit(10);
		char ok = 1;
		if (write(notify[1], &ok, 1) != 1)
			_exit(11);
		for (;;)
			pause();
	}
	close(notify[1]);
	char ok = 0;
	REQUIRE(read(notify[0], &ok, 1) == 1);
	close(notify[0]);

	REQUIRE(kill(pid, SIGKILL) == 0);
	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGKILL);

	// EOWNERDEAD -> refuse with EBUSY and latch ENOTRECOVERABLE.
	REQUIRE_FALSE(m->lock("test"));
	REQUIRE(errno == EBUSY);
	// Every later caller fails the same way; half-mutated state is never
	// touched again.
	REQUIRE_FALSE(m->lock("test"));
	REQUIRE(errno == EBUSY);

	munmap(m, sizeof(*m));
}
#endif
