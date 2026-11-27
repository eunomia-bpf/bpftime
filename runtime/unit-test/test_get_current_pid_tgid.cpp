#include "catch2/catch_test_macros.hpp"
#include "bpftime_config.hpp"
#include <cstdint>
#include <sys/wait.h>
#include <unistd.h>

#if __APPLE__
#include <pthread.h>
#else
#include <sys/syscall.h>
#endif

extern "C" {
uint64_t bpftime_get_current_pid_tgid(uint64_t, uint64_t, uint64_t, uint64_t,
				      uint64_t);
}

static uint64_t call_helper()
{
	return bpftime_get_current_pid_tgid(0, 0, 0, 0, 0);
}

static uint32_t helper_tgid(uint64_t v)
{
	return static_cast<uint32_t>(v >> 32);
}

static uint32_t helper_tid(uint64_t v)
{
	return static_cast<uint32_t>(v);
}

// Mirror the identity the helper is expected to report on this platform.
static uint32_t current_tid()
{
#if __APPLE__
	uint64_t tid = 0;
	pthread_threadid_np(NULL, &tid);
	return static_cast<uint32_t>(tid);
#else
	return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}

static bool helper_matches_this_process()
{
	uint64_t value = call_helper();
	return helper_tgid(value) == static_cast<uint32_t>(getpid()) &&
	       helper_tid(value) == current_tid();
}

// The helper caches the calling identity. A forked child inherits that cache,
// so without refreshing it in the child the helper keeps reporting the
// parent's pid and tid instead of the child's own.
TEST_CASE("bpf_get_current_pid_tgid reports the calling process after fork")
{
	REQUIRE(helper_matches_this_process());

	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		bool ok = helper_matches_this_process();
		// A grandchild must also observe its own identity, not the
		// value inherited through two levels of fork.
		pid_t grandchild = fork();
		if (grandchild < 0) {
			_exit(1);
		}
		if (grandchild == 0) {
			_exit(helper_matches_this_process() ? 0 : 1);
		}
		int gstatus = 0;
		bool grandchild_ok = waitpid(grandchild, &gstatus, 0) ==
					     grandchild &&
				     WIFEXITED(gstatus) &&
				     WEXITSTATUS(gstatus) == 0;
		_exit((ok && grandchild_ok) ? 0 : 1);
	}

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);

	// The parent's own cached identity must be unaffected by the children.
	REQUIRE(helper_matches_this_process());
}

#if __linux__
#include <boost/scope_exit.hpp>
#include <filesystem>
#include <fstream>
#include <thread>
#include <linux/bpf_common.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>

TEST_CASE("PID namespace identity translates on cache fill and refreshes on fork")
{
	char dir[] = "/tmp/bpftime-nspid-XXXXXX";
	REQUIRE(mkdtemp(dir) != nullptr);
	const std::filesystem::path root(dir);
	BOOST_SCOPE_EXIT_ALL(=) { std::filesystem::remove_all(root); };
	std::filesystem::create_directory(root / "self");
	std::filesystem::create_directory(root / "thread-self");
	{
		std::ofstream status(root / "self/status");
		status << "Groups:";
		for (int i = 0; i < 1024; ++i)
			status << " 12345";
		status << "\nNSpid:\t40001\t1\n";
	}
	std::ofstream(root / "thread-self/status") << "NSpid:\t40002\t2\n";
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		if (setenv("BPFTIME_HOST_PROCFS", dir, 1) != 0)
			_exit(1);
		bpftime::reload_agent_bootstrap_config_from_env();
		const uint64_t first = (uint64_t(40001) << 32) | 40002;
		if (call_helper() != first)
			_exit(2);
		std::ofstream(root / "self/status") << "NSpid:\t50001\t1\n";
		std::ofstream(root / "thread-self/status") << "NSpid:\t50002\t2\n";
		if (call_helper() != first)
			_exit(3);
		const uint64_t next = (uint64_t(50001) << 32) | 50002;
		uint64_t threaded = 0;
		std::thread worker([&] { threaded = call_helper(); });
		worker.join();
		if (threaded != next)
			_exit(4);
		pid_t grandchild = fork();
		if (grandchild == 0)
			_exit(call_helper() == next ? 0 : 5);
		int status = 0;
		_exit(grandchild > 0 && waitpid(grandchild, &status, 0) == grandchild &&
		      WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 6);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("PID namespace identity falls back when procfs is unavailable")
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		setenv("BPFTIME_HOST_PROCFS", "/nonexistent/bpftime-procfs", 1);
		bpftime::reload_agent_bootstrap_config_from_env();
		_exit(helper_matches_this_process() ? 0 : 1);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("PID namespace identity hot path makes no system calls")
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		const uint64_t expected = call_helper();
		// After warming the cache, permit only process exit. Any helper
		// syscall (including libc getpid) terminates this test child.
		// libbpf ships an internal linux/filter.h, so describe the
		// classic BPF UAPI layout here instead of including that header.
		struct instruction { uint16_t code; uint8_t jt, jf; uint32_t k; };
		instruction filter[] = {
			{ BPF_LD | BPF_W | BPF_ABS, 0, 0, offsetof(struct seccomp_data, nr) },
			{ BPF_JMP | BPF_JEQ | BPF_K, 2, 0, SYS_exit_group },
			{ BPF_JMP | BPF_JEQ | BPF_K, 1, 0, SYS_exit },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL_PROCESS },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW },
		};
		struct { unsigned short len; instruction *filter; } program = { 5, filter };
		if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
		    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0)
			_exit(2);
		for (int i = 0; i < 10000; ++i)
			if (call_helper() != expected)
				_exit(3);
		_exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

#include <cerrno>
#include <sys/stat.h>

extern "C" uint64_t bpftime_get_ns_current_pid_tgid(uint64_t, uint64_t,
						    uint64_t, uint64_t,
						    uint64_t);

TEST_CASE("bpf_get_ns_current_pid_tgid answers for the own PID namespace")
{
	struct stat st {};
	REQUIRE(stat("/proc/self/ns/pid", &st) == 0);
	struct {
		uint32_t pid;
		uint32_t tgid;
	} info{ 1, 1 };
	const auto call = [&](uint64_t dev, uint64_t ino, uint64_t size) {
		return (int64_t)bpftime_get_ns_current_pid_tgid(
			dev, ino, (uint64_t)(uintptr_t)&info, size, 0);
	};
	REQUIRE(call(st.st_dev, st.st_ino, sizeof(info)) == 0);
	REQUIRE(info.tgid == (uint32_t)getpid());
	REQUIRE(info.pid == (uint32_t)syscall(SYS_gettid));
	// Another namespace: -EINVAL and zeroed output, as in the kernel.
	REQUIRE(call(st.st_dev, st.st_ino + 1, sizeof(info)) == -EINVAL);
	REQUIRE(info.pid == 0);
	REQUIRE(info.tgid == 0);
	REQUIRE(call(st.st_dev, st.st_ino, sizeof(info) - 1) == -EINVAL);
}

TEST_CASE("bpf_get_ns_current_pid_tgid caches a failed /proc/self/ns/pid stat")
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		struct instruction { uint16_t code; uint8_t jt, jf; uint32_t k; };
		instruction fail_stat[16];
		unsigned short n = 0;
		fail_stat[n++] = { BPF_LD | BPF_W | BPF_ABS, 0, 0,
				   offsetof(struct seccomp_data, nr) };
		auto deny = [&](uint32_t nr) {
			fail_stat[n++] = { BPF_JMP | BPF_JEQ | BPF_K, 0, 1, nr };
			fail_stat[n++] = { BPF_RET | BPF_K, 0, 0,
					   SECCOMP_RET_ERRNO | ENOENT };
		};
#ifdef SYS_stat
		deny(SYS_stat);
#endif
#ifdef SYS_newfstatat
		deny(SYS_newfstatat);
#endif
#ifdef SYS_fstatat64
		deny(SYS_fstatat64);
#endif
#ifdef SYS_statx
		deny(SYS_statx);
#endif
		fail_stat[n++] = { BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW };
		struct {
			unsigned short len;
			instruction *filter;
		} prog1 = { n, fail_stat };
		if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
		    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog1) != 0)
			_exit(2);

		struct {
			uint32_t pid;
			uint32_t tgid;
		} info{ 7, 9 };
		errno = EAGAIN;
		int64_t rc = (int64_t)bpftime_get_ns_current_pid_tgid(
			0, UINT64_MAX, (uint64_t)(uintptr_t)&info, sizeof(info),
			0);
		if (rc != -EINVAL || errno != EAGAIN || info.pid != 0 ||
		    info.tgid != 0)
			_exit(3);

		// Stack a second filter killing on any syscall other than exit:
		// proves the failed stat sentinel prevents all further syscalls.
		instruction kill_all[] = {
			{ BPF_LD | BPF_W | BPF_ABS, 0, 0,
			  offsetof(struct seccomp_data, nr) },
			{ BPF_JMP | BPF_JEQ | BPF_K, 2, 0, SYS_exit_group },
			{ BPF_JMP | BPF_JEQ | BPF_K, 1, 0, SYS_exit },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL_PROCESS },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW },
		};
		struct {
			unsigned short len;
			instruction *filter;
		} prog2 = { 5, kill_all };
		if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog2) != 0)
			_exit(4);
		for (int i = 0; i < 1000; ++i) {
			info = { 5, 6 };
			rc = (int64_t)bpftime_get_ns_current_pid_tgid(
				0, (i & 1) ? UINT64_MAX : 1,
				(uint64_t)(uintptr_t)&info, sizeof(info), 0);
			if (rc != -EINVAL || info.pid != 0 || info.tgid != 0)
				_exit(5);
		}
		_exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
#endif
