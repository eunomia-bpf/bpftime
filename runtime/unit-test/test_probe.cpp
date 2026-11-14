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
#include <cstdlib>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <atomic>
#include <vector>
#include <unistd.h>
#include <signal.h>
#include <atomic>
#include <bpftime_helper_group.hpp>
#include <linux/filter.h>
#if defined(__linux__)
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif

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
int64_t bpftime_probe_read_user(uint64_t, int64_t, uint64_t, uint64_t,
				uint64_t);
int64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, int64_t len,
				 uint64_t, uint64_t);
int64_t bpf_probe_read_str(uint64_t dst, uint64_t size, uint64_t src, uint64_t,
			   uint64_t);
}

#if defined(BPFTIME_HAVE_FRIDA) && defined(__linux__) &&                       \
	(defined(__x86_64__) || defined(__aarch64__))
// Isolate process-wide signal disposition and once-only initialization from
// the other probe tests. Only the parent executes Catch assertions.
template <typename F> static int run_probe_read_child(F body)
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		struct rlimit no_core = { 0, 0 };
		setrlimit(RLIMIT_CORE, &no_core);
		alarm(20);
		try {
			_exit(body());
		} catch (...) {
			_exit(127);
		}
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	return status;
}

TEST_CASE("Probe read user recovers memory faults", "[probe-read-user]")
{
	int status = run_probe_read_child([] {
		int source = 42, destination = 0;
		if (bpftime_probe_read_user(0, 0, 0, 0, 0) != 0 ||
		    bpftime_probe_read_user(0, -1, 0, 0, 0) != -EFAULT)
			return 1;
		long page_size = sysconf(_SC_PAGESIZE);
		if (page_size <= 0)
			return 2;
		auto *pages = static_cast<char *>(
			mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
		if (pages == MAP_FAILED ||
		    mprotect(pages + page_size, page_size, PROT_NONE) != 0)
			return 3;
		char buffer[16] = {};
		sigset_t blocked;
		sigemptyset(&blocked);
		sigaddset(&blocked, SIGUSR1);
		if (pthread_sigmask(SIG_BLOCK, &blocked, nullptr) != 0)
			return 9;
		for (int i = 0; i < 100; ++i) {
			errno = EDOM;
			if (bpftime_probe_read_user((uint64_t)&destination,
						    sizeof(int), 0, 0,
						    0) != -EFAULT ||
			    errno != EDOM)
				return 4;
			if (bpftime_probe_read_user(0, sizeof(int),
						    (uint64_t)&source, 0,
						    0) != -EFAULT)
				return 5;
			if (bpftime_probe_read_user(
				    (uint64_t)buffer, sizeof(buffer),
				    (uint64_t)(pages + page_size - 8), 0,
				    0) != -EFAULT)
				return 6;
			if (bpftime_probe_read_user(
				    (uint64_t)&destination, sizeof(int),
				    (uint64_t)&source, 0, 0) != 0 ||
			    destination != source)
				return 7;
			sigset_t mask;
			if (pthread_sigmask(SIG_SETMASK, nullptr, &mask) != 0 ||
			    sigismember(&mask, SIGSEGV) != 0 ||
			    sigismember(&mask, SIGUSR1) != 1)
				return 8;
		}
		munmap(pages, page_size * 2);
		return 0;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	REQUIRE_FALSE(bpftime::helper_is_async_signal_safe(
		112, (void *)bpftime_probe_read_user));
}

TEST_CASE("Successful probe copies make no syscalls after initialization",
	  "[probe-read-user]")
{
	int status = run_probe_read_child([] {
		int source = 42, destination = 0;
		if (bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination),
					    (uint64_t)&source, 0, 0) != 0)
			return 1;
		// Allow only process exit after warming up. Any syscall made by
		// a subsequent copy kills the child, including hidden libc
		// calls.
		// The repository's linux/filter.h supplies eBPF macros rather
		// than the classic BPF records consumed by seccomp.
		struct filter_insn {
			uint16_t code;
			uint8_t jt, jf;
			uint32_t k;
		};
		filter_insn filter[] = {
			{ BPF_LD | BPF_W | BPF_ABS, 0, 0, 0 },
			{ BPF_JMP | BPF_JEQ | BPF_K, 2, 0, SYS_exit },
			{ BPF_JMP | BPF_JEQ | BPF_K, 1, 0, SYS_exit_group },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL_PROCESS },
			{ BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW },
		};
		struct filter_program {
			unsigned short len;
			filter_insn *filter;
		};
		filter_program program = { (unsigned short)(sizeof(filter) /
							    sizeof(filter[0])),
					   filter };
		if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
		    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0)
			return 2;
		for (int i = 0; i < 1000; ++i) {
			destination = 0;
			if (bpftime_probe_read_user(
				    (uint64_t)&destination, sizeof(destination),
				    (uint64_t)&source, 0, 0) != 0 ||
			    destination != source)
				return 3;
#ifndef ENABLE_PROBE_WRITE_CHECK
			destination = 0;
			if (bpftime_probe_write_user(
				    (uint64_t)&destination, (uint64_t)&source,
				    sizeof(source), 0, 0) != 0 ||
			    destination != source)
				return 4;
#endif
		}
		return 0;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("Probe read user isolates concurrent recovery", "[probe-read-user]")
{
	int status = run_probe_read_child([] {
		std::atomic<bool> failed = false;
		std::thread workers[8];
		for (auto &worker : workers)
			worker = std::thread([&] {
				int src = 17, dst = 0;
				for (int i = 0; i < 100; ++i) {
					if (bpftime_probe_read_user(
						    (uint64_t)&dst, sizeof(dst),
						    0, 0, 0) != -EFAULT ||
					    bpftime_probe_read_user(
						    (uint64_t)&dst, sizeof(dst),
						    (uint64_t)&src, 0,
						    0) != 0 ||
					    dst != src)
						failed = true;
					if (bpftime_probe_write_user(
						    0, (uint64_t)&src,
						    sizeof(src), 0,
						    0) != -EFAULT ||
					    bpftime_probe_write_user(
						    (uint64_t)&dst,
						    (uint64_t)&src, sizeof(src),
						    0, 0) != 0 ||
					    dst != src)
						failed = true;
				}
			});
		for (auto &worker : workers)
			worker.join();
		return failed ? 1 : 0;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("Probe write user handles invalid memory",
	  "[probe-read-user][probe-write-user]")
{
	int status = run_probe_read_child([] {
		int source = 42, destination = 0;
		if (bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination),
					    (uint64_t)&source, 0, 0) != 0)
			return 1;
		struct sigaction before = {}, after = {};
		if (sigaction(SIGSEGV, nullptr, &before) != 0)
			return 2;
		long page_size = sysconf(_SC_PAGESIZE);
		if (page_size <= 0)
			return 3;
		auto *pages = static_cast<char *>(
			mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
		if (pages == MAP_FAILED ||
		    mprotect(pages + page_size, page_size, PROT_NONE) != 0)
			return 4;
		char buffer[16] = {};
		for (int i = 0; i < 100; ++i) {
			errno = EDOM;
			if (bpftime_probe_write_user(0, (uint64_t)&source,
						     sizeof(source), 0,
						     0) != -EFAULT)
				return 5;
#ifndef ENABLE_PROBE_WRITE_CHECK
			if (errno != EDOM)
				return 5;
#endif
			if (bpftime_probe_write_user((uint64_t)&destination, 0,
						     sizeof(source), 0,
						     0) != -EFAULT)
				return 6;
			if (bpftime_probe_write_user(
				    (uint64_t)(pages + page_size - 8),
				    (uint64_t)buffer, sizeof(buffer), 0,
				    0) != -EFAULT)
				return 7;
			if (bpftime_probe_write_user(
				    (uint64_t)&destination, (uint64_t)&source,
				    sizeof(source), 0, 0) != 0 ||
			    destination != source)
				return 8;
		}
		munmap(pages, page_size * 2);
		if (sigaction(SIGSEGV, nullptr, &after) != 0 ||
		    after.sa_sigaction != before.sa_sigaction ||
		    after.sa_flags != before.sa_flags)
			return 9;
		return bpftime_probe_write_user(0, 0, 0, 0, 0) == 0 &&
				       bpftime_probe_write_user(0, 0, -1, 0,
								0) == -EFAULT ?
			       0 :
			       10;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	REQUIRE_FALSE(bpftime::helper_is_async_signal_safe(
		36, (void *)bpftime_probe_write_user));
}

TEST_CASE("Probe access falls back with blocked SIGSEGV", "[probe-read-user]")
{
	int status = run_probe_read_child([] {
		sigset_t blocked;
		sigemptyset(&blocked);
		sigaddset(&blocked, SIGSEGV);
		if (pthread_sigmask(SIG_BLOCK, &blocked, nullptr) != 0)
			return 1;
		int source = 42, destination = 0;
		if (bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination),
					    (uint64_t)&source, 0, 0) != 0 ||
		    destination != source)
			return 2;
		destination = 0;
		if (bpftime_probe_write_user((uint64_t)&destination,
					     (uint64_t)&source, sizeof(source),
					     0, 0) != 0 ||
		    destination != source)
			return 3;
#ifdef ENABLE_PROBE_READ_CHECK
		if (bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination), 0, 0,
					    0) != -EFAULT)
			return 4;
#endif
#ifdef ENABLE_PROBE_WRITE_CHECK
		if (bpftime_probe_write_user(0, (uint64_t)&destination,
					     sizeof(destination), 0,
					     0) != -EFAULT)
			return 5;
#endif
		sigset_t current;
		if (pthread_sigmask(SIG_SETMASK, nullptr, &current) != 0 ||
		    sigismember(&current, SIGSEGV) != 1)
			return 6;
		return 0;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("Probe access does not write host stdout or stderr",
	  "[probe-read-user]")
{
	int output[2], errors[2];
	REQUIRE(pipe(output) == 0);
	REQUIRE(pipe(errors) == 0);
	int status = run_probe_read_child([&] {
		close(output[0]);
		close(errors[0]);
		if (dup2(output[1], STDOUT_FILENO) < 0 ||
		    dup2(errors[1], STDERR_FILENO) < 0)
			return 1;
		close(output[1]);
		close(errors[1]);
		int source = 3, destination = 0;
		if (bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination),
					    (uint64_t)&source, 0, 0) != 0 ||
		    bpftime_probe_read_user((uint64_t)&destination,
					    sizeof(destination), 0, 0,
					    0) != -EFAULT ||
		    bpftime_probe_write_user((uint64_t)&destination,
					     (uint64_t)&source, sizeof(source),
					     0, 0) != 0 ||
		    bpftime_probe_write_user(0, (uint64_t)&source,
					     sizeof(source), 0, 0) != -EFAULT)
			return 2;
		return 0;
	});
	close(output[1]);
	close(errors[1]);
	char buffer[64];
	auto output_size = read(output[0], buffer, sizeof(buffer));
	auto error_size = read(errors[0], buffer, sizeof(buffer));
	close(output[0]);
	close(errors[0]);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	REQUIRE(output_size == 0);
	REQUIRE(error_size == 0);
}

TEST_CASE("Probe read user leaves unrelated SIGSEGV fatal", "[probe-read-user]")
{
	for (bool explicit_signal : { false, true }) {
		int status = run_probe_read_child([=] {
			int src = 1, dst = 0;
			if (bpftime_probe_read_user((uint64_t)&dst, sizeof(dst),
						    (uint64_t)&src, 0, 0) != 0)
				return 2;
			std::thread worker([=] {
				if (explicit_signal)
					raise(SIGSEGV);
				else {
					void *page = mmap(
						nullptr, 4096, PROT_NONE,
						MAP_PRIVATE | MAP_ANONYMOUS, -1,
						0);
					*static_cast<volatile char *>(page) = 1;
				}
			});
			worker.join();
			return 3;
		});
		REQUIRE(WIFSIGNALED(status));
		REQUIRE(WTERMSIG(status) == SIGSEGV);
	}
}

TEST_CASE("Probe access leaves replacement handler installed",
	  "[probe-read-user]")
{
	int status = run_probe_read_child([] {
		int src = 1, dst = 0;
		if (bpftime_probe_read_user((uint64_t)&dst, sizeof(dst),
					    (uint64_t)&src, 0, 0) != 0)
			return 1;
		struct sigaction action = {};
		action.sa_handler = count_sigsegv;
		sigemptyset(&action.sa_mask);
		if (sigaction(SIGSEGV, &action, nullptr) != 0)
			return 2;
		dst = 0;
		if (bpftime_probe_read_user((uint64_t)&dst, sizeof(dst),
					    (uint64_t)&src, 0, 0) != 0 ||
		    dst != src)
			return 3;
		dst = 0;
		if (bpftime_probe_write_user((uint64_t)&dst, (uint64_t)&src,
					     sizeof(src), 0, 0) != 0 ||
		    dst != src)
			return 4;
		struct sigaction current = {};
		if (sigaction(SIGSEGV, nullptr, &current) != 0 ||
		    current.sa_handler != count_sigsegv)
			return 6;
		raise(SIGSEGV);
		return forwarded_sigsegv == 1 ? 0 : 7;
	});
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("BPF helper 112 invokes probe read user", "[probe-read-user]")
{
	for (bool jit : { false, true }) {
		int status = run_probe_read_child([=] {
			struct bpf_insn instructions[] = {
				BPF_MOV64_REG(1, 10),
				BPF_ALU64_IMM(BPF_ADD, 1, -8),
				BPF_MOV64_IMM(2, 4),
				BPF_MOV64_IMM(3, 0),
				BPF_MOV64_IMM(4, 0),
				BPF_MOV64_IMM(5, 0),
				BPF_EMIT_CALL(112),
				BPF_EXIT_INSN(),
			};
			bpftime::runtime_config config;
#if BPFTIME_LLVM_JIT
			config.set_vm_name("llvm");
#else
			config.set_vm_name("ubpf");
#endif
			bpftime::bpftime_prog prog(
				(const ebpf_inst *)instructions,
				sizeof(instructions) / sizeof(instructions[0]),
				"probe_read_user", config);
			if (bpftime::bpftime_helper_group::
					    get_kernel_utils_helper_group()
						    .add_helper_group_to_prog(
							    &prog) != 0 ||
			    prog.bpftime_prog_load(jit) != 0)
				return 1;
			uint64_t result = 0;
			int context = 0;
			if (prog.bpftime_prog_exec(&context, sizeof(context),
						   &result) != 0)
				return 2;
			struct sigaction action = {};
			if (sigaction(SIGSEGV, nullptr, &action) != 0 ||
			    !(action.sa_flags & SA_SIGINFO))
				return 3;
			// Unlike the old process_vm_readv implementation, the
			// new helper installs its handler. Recovery must also
			// work below a JIT frame.
			return (int64_t)result == -EFAULT ? 0 : 4;
		});
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
	}
}
#endif

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

#if defined(__linux__)
TEST_CASE("Probe read user preserves default host signal dispositions", "[probe-default-signal-disposition]")
{
	// exec gives this check fresh once-flags even when other probe tests ran
	// first; fork alone would inherit the initialized guard from its parent.
	if (getenv("BPFTIME_PROBE_DISPOSITION_TEST_CHILD")) {
		struct sigaction host = {}, before_segv = {}, before_bus = {};
		host.sa_handler = count_sigsegv;
		host.sa_flags = SA_RESTART;
		sigemptyset(&host.sa_mask);
		sigaddset(&host.sa_mask, SIGUSR1);
		if (sigaction(SIGSEGV, &host, nullptr) != 0)
			_exit(1);
		host.sa_handler = SIG_IGN;
		if (sigaction(SIGBUS, &host, nullptr) != 0 ||
		    sigaction(SIGSEGV, nullptr, &before_segv) != 0 ||
		    sigaction(SIGBUS, nullptr, &before_bus) != 0)
			_exit(2);
		int source = 42, destination = 0;
		if (bpftime_probe_read_user((uint64_t)&destination, sizeof(destination),
			   (uint64_t)&source, 0, 0) != 0 || destination != source)
			_exit(3);
		struct sigaction after_segv = {}, after_bus = {};
		if (sigaction(SIGSEGV, nullptr, &after_segv) != 0 ||
		    sigaction(SIGBUS, nullptr, &after_bus) != 0 ||
		    after_segv.sa_handler != before_segv.sa_handler ||
		    after_segv.sa_flags != before_segv.sa_flags ||
		    sigismember(&after_segv.sa_mask, SIGUSR1) != 1 ||
		    after_bus.sa_handler != before_bus.sa_handler ||
		    after_bus.sa_flags != before_bus.sa_flags)
			_exit(4);
		_exit(0);
	}
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		if (setenv("BPFTIME_PROBE_DISPOSITION_TEST_CHILD", "1", 1) != 0)
			_exit(5);
		execl("/proc/self/exe", "probe-disposition-test",
		      "[probe-default-signal-disposition]", (char *)nullptr);
		_exit(6);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
#endif

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
	uint64_t src[4] = { 1, 2, 3, 4 };
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
	uint64_t src[4] = { 1, 2, 3, 4 };
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
	uint64_t src[2] = { 1, 2 };
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
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
