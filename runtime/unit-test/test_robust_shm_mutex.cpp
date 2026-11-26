/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#if defined(__linux__)
#include "robust_shm_mutex.hpp"
#include <atomic>
#include <cerrno>
#include <new>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct shared_test_region {
	bpftime::robust_shm_mutex lock_taint;
	bpftime::robust_shm_mutex lock_recover;
	std::atomic<int> child_inside{ 0 };
	std::atomic<int> stolen_while_held{ 0 };
};

bpftime::robust_shm_mutex *make_shared_mutex()
{
	void *p = mmap(nullptr, sizeof(bpftime::robust_shm_mutex),
		       PROT_READ | PROT_WRITE,
		       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	REQUIRE(p != MAP_FAILED);
	return new (p) bpftime::robust_shm_mutex();
}

pid_t fork_killed_holder(bpftime::robust_shm_mutex *m, bool reap_immediately)
{
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
		close(notify[1]);
		if (!reap_immediately)
			raise(SIGKILL);
		for (;;)
			pause();
	}
	close(notify[1]);
	char ok = 0;
	REQUIRE(read(notify[0], &ok, 1) == 1);
	close(notify[0]);
	if (reap_immediately) {
		REQUIRE(kill(pid, SIGKILL) == 0);
		int status = 0;
		REQUIRE(waitpid(pid, &status, 0) == pid);
		REQUIRE(WIFSIGNALED(status));
		REQUIRE(WTERMSIG(status) == SIGKILL);
	}
	return pid;
}
} // namespace

TEST_CASE("robust SHM mutex taints permanently after a SIGKILLed holder",
	  "[robust_shm_mutex]")
{
	auto *m = make_shared_mutex();
	(void)fork_killed_holder(m, true);

	REQUIRE_FALSE(m->lock("test"));
	REQUIRE(errno == EBUSY);
	REQUIRE_FALSE(m->lock("test"));
	REQUIRE(errno == EBUSY);

	munmap(m, sizeof(*m));
}

TEST_CASE("robust SHM mutex recovers an unreaped zombie SIGKILLed holder",
	  "[robust_shm_mutex]")
{
	auto *m = make_shared_mutex();
	pid_t pid = fork_killed_holder(m, false);

	REQUIRE(m->lock_and_recover());
	m->unlock();
	REQUIRE(m->lock_and_recover());
	m->unlock();

	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFSIGNALED(status));
	munmap(m, sizeof(*m));
}

TEST_CASE("robust SHM mutex waits on live holder in a child PID namespace",
	  "[robust_shm_mutex]")
{
	void *raw = mmap(nullptr, sizeof(shared_test_region),
			 PROT_READ | PROT_WRITE,
			 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	REQUIRE(raw != MAP_FAILED);
	auto *reg = new (raw) shared_test_region();

	int ready_pipe[2]{ -1, -1 };
	REQUIRE(pipe(ready_pipe) == 0);
	pid_t outer = fork();
	REQUIRE(outer >= 0);
	if (outer == 0) {
		close(ready_pipe[0]);
		if (unshare(CLONE_NEWPID) != 0 &&
		    unshare(CLONE_NEWUSER | CLONE_NEWPID) != 0) {
			char skip = 's';
			(void)write(ready_pipe[1], &skip, 1);
			close(ready_pipe[1]);
			_exit(0);
		}
		pid_t inner = fork();
		if (inner < 0)
			_exit(20);
		if (inner == 0) {
			if (!reg->lock_taint.lock("child-pidns") ||
			    !reg->lock_recover.lock_and_recover())
				_exit(21);
			reg->child_inside.store(1, std::memory_order_release);
			char ok = 'r';
			(void)write(ready_pipe[1], &ok, 1);
			close(ready_pipe[1]);
			// Hold across a child PID namespace so the parent blocks in futex wait
			usleep(20000);
			reg->child_inside.store(0, std::memory_order_release);
			reg->lock_recover.unlock();
			reg->lock_taint.unlock();
			_exit(0);
		}
		close(ready_pipe[1]);
		int st = 0;
		if (waitpid(inner, &st, 0) != inner || !WIFEXITED(st))
			_exit(22);
		_exit(WEXITSTATUS(st));
	}
	close(ready_pipe[1]);
	char sig = 0;
	REQUIRE(read(ready_pipe[0], &sig, 1) == 1);
	close(ready_pipe[0]);
	if (sig == 's') {
		int st = 0;
		waitpid(outer, &st, 0);
		munmap(reg, sizeof(*reg));
		SKIP("CLONE_NEWPID unavailable in this container");
	}
	REQUIRE(sig == 'r');

	// Parent waits on both lock() and lock_and_recover():
	// neither may taint or steal while the child PID-namespace holder is inside.
	REQUIRE(reg->lock_taint.lock("parent-wait"));
	if (reg->child_inside.load(std::memory_order_acquire) != 0)
		reg->stolen_while_held.store(1, std::memory_order_relaxed);
	reg->lock_taint.unlock();

	REQUIRE(reg->lock_recover.lock_and_recover());
	if (reg->child_inside.load(std::memory_order_acquire) != 0)
		reg->stolen_while_held.store(1, std::memory_order_relaxed);
	reg->lock_recover.unlock();

	int outer_status = 0;
	REQUIRE(waitpid(outer, &outer_status, 0) == outer);
	REQUIRE(WIFEXITED(outer_status));
	REQUIRE(WEXITSTATUS(outer_status) == 0);
	REQUIRE(reg->stolen_while_held.load() == 0);

	munmap(reg, sizeof(*reg));
}
#endif
