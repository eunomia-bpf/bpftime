/* SPDX-License-Identifier: MIT */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#if defined(__x86_64__) || defined(__aarch64__)
#include "synthetic_kallsyms.hpp"
#endif
#include "syscall_context.hpp"
#include "syscall_server_utils.hpp"
#include <boost/scope_exit.hpp>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <linux/bpf.h>
#include <linux/stat.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct shm_fixture {
	shm_fixture()
	{
		setenv("BPFTIME_GLOBAL_SHM_NAME",
		       ("bpftime-mock-fd-test-" +
			std::to_string(getpid()))
			       .c_str(),
		       1);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	}
	~shm_fixture()
	{
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_NO_CREATE);
		unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	}
};
#if defined(__x86_64__) || defined(__aarch64__)
struct analysis_value {
	uint64_t address;
	uint32_t pid;
	int32_t error;
	uint8_t code[128];
};
#endif

constexpr const char *UPROBE_TYPE_VIRTUAL_PATH =
	"/sys/bus/event_source/devices/uprobe/type";
constexpr const char *KPROBE_TYPE_VIRTUAL_PATH =
	"/sys/bus/event_source/devices/kprobe/type";
} // namespace

TEST_CASE("cached real-file synthetic files and synthetic tpbase completion",
	  "[syscall-server][mock-file][system-analysis]")
{
	shm_fixture fixture;
	syscall_context ctx;
	const std::string expected =
		std::to_string(MOCKED_UPROBE_TYPE_VALUE);
	int fd = ctx.handle_openat(
		AT_FDCWD, UPROBE_TYPE_VIRTUAL_PATH, O_RDONLY, 0);
	REQUIRE(fd >= 0);
	// A bare SYS_read (no libc interposition) reads the cached real file
	// through the ordinary kernel file path.
	char raw[32] = {};
	REQUIRE(syscall(SYS_read, fd, raw, sizeof(raw)) ==
		(long)expected.size());
	REQUIRE(expected == raw);
	// One kernel cursor: rewind makes the payload readable again.
	REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
	REQUIRE(syscall(SYS_read, fd, raw, sizeof(raw)) ==
		(long)expected.size());
	REQUIRE(ctx.handle_close(fd) == 0);
#if defined(__x86_64__) || defined(__aarch64__)
	ctx.try_startup();
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");
	bpftime::bpf_map_attr attr{
		.type = BPF_MAP_TYPE_ARRAY, .key_size = sizeof(uint32_t),
		.value_size = sizeof(analysis_value), .max_ents = 1
	};
	int map_fd = bpftime_maps_create(-1, "system_analysis", attr);
	REQUIRE(map_fd >= 0);
#if defined(__x86_64__)
	constexpr uint64_t addr =
		bpftime::SYNTHETIC_X86_FSBASE_WRITE_TASK_ADDR;
#elif defined(__aarch64__)
	constexpr uint64_t addr = bpftime::SYNTHETIC_ARM64_TLS_SET_ADDR;
#endif
	analysis_value value{ .address = addr, .pid = 4321 };
	uint32_t key = 0;
	REQUIRE(bpftime_map_update_elem(map_fd, &key, &value, 0) == 0);
	analysis_value result{};
	union bpf_attr lookup{
		.map_fd = (uint32_t)map_fd, .key = (uint64_t)&key,
		.value = (uint64_t)&result
	};
	REQUIRE(ctx.handle_sysbpf(BPF_MAP_LOOKUP_ELEM, &lookup,
				  sizeof(lookup)) == 0);
	REQUIRE(result.pid == 0);
	REQUIRE(result.address == 0);
	uint32_t off = 0;
#if defined(__x86_64__)
	REQUIRE(result.code[0] == 0x48);
	REQUIRE(result.code[1] == 0x89);
	REQUIRE(result.code[2] == 0xb7);
	memcpy(&off, &result.code[3], sizeof(off));
#elif defined(__aarch64__)
	uint32_t insn = 0;
	memcpy(&insn, result.code, sizeof(insn));
	REQUIRE(insn == (0xf9400000U |
			 (((bpftime_get_current_task_tpbase_offset() / 8U) &
			   0xfffU)
			  << 10)));
	off = (uint32_t)bpftime_get_current_task_tpbase_offset();
#endif
	REQUIRE(off == (uint32_t)bpftime_get_current_task_tpbase_offset());
#endif
}

TEST_CASE("synthetic files resolve to cached read-only real files",
	  "[syscall-server][mock-file][synthetic-fs]")
{
	shm_fixture fixture;
	syscall_context ctx;
	const std::string expected =
		std::to_string(MOCKED_UPROBE_TYPE_VALUE);

	// Resolution materializes the backing file once inside a private
	// directory and keeps returning the same host path.
	auto first = bpftime::resolve_redirected_path(
		UPROBE_TYPE_VIRTUAL_PATH);
	REQUIRE(first.has_value());
	auto second = bpftime::resolve_redirected_path(
		UPROBE_TYPE_VIRTUAL_PATH);
	REQUIRE(second.has_value());
	REQUIRE(*first == *second);
	const auto synthetic_dir =
		std::filesystem::path(*first).parent_path();
	size_t entries = 0;
	for (const auto &entry :
	     std::filesystem::directory_iterator(synthetic_dir))
		++entries;
	REQUIRE(entries == 1);
	struct stat backing_st = {};
	REQUIRE(stat(first->c_str(), &backing_st) == 0);
	REQUIRE((backing_st.st_mode & 0777) == 0400);
	REQUIRE(S_ISREG(backing_st.st_mode));

	// Every open family reaches the same real file, and the returned
	// descriptor supports the ordinary kernel file operations.
	int at_fd = ctx.handle_openat(AT_FDCWD,
				      UPROBE_TYPE_VIRTUAL_PATH, O_RDONLY, 0);
	REQUIRE(at_fd >= 0);
	char buffer[32] = {};
	REQUIRE(pread(at_fd, buffer, expected.size(), 0) ==
		(ssize_t)expected.size());
	REQUIRE(expected == buffer);
	struct stat fd_st = {};
	REQUIRE(fstat(at_fd, &fd_st) == 0);
	REQUIRE(fd_st.st_size == (off_t)expected.size());
	void *mapping = mmap(nullptr, expected.size(), PROT_READ,
			     MAP_PRIVATE, at_fd, 0);
	REQUIRE(mapping != MAP_FAILED);
	REQUIRE(memcmp(mapping, expected.data(), expected.size()) == 0);
	REQUIRE(munmap(mapping, expected.size()) == 0);
	REQUIRE(ctx.handle_close(at_fd) == 0);

	int plain_fd = ctx.handle_open(UPROBE_TYPE_VIRTUAL_PATH, O_RDONLY, 0);
	REQUIRE(plain_fd >= 0);
	memset(buffer, 0, sizeof(buffer));
	REQUIRE(read(plain_fd, buffer, sizeof(buffer)) ==
		(ssize_t)expected.size());
	REQUIRE(expected == buffer);
	REQUIRE(ctx.handle_close(plain_fd) == 0);

	FILE *fp = ctx.handle_fopen(UPROBE_TYPE_VIRTUAL_PATH, "re");
	REQUIRE(fp != nullptr);
	REQUIRE(fileno(fp) >= 0);
	char fopen_buffer[32] = {};
	REQUIRE(fread(fopen_buffer, 1, expected.size(), fp) ==
		expected.size());
	REQUIRE(expected == fopen_buffer);
	REQUIRE(fclose(fp) == 0);

	// A raw syscall open/read pair (bypassing libc interposition) works
	// against the materialized host path as well.
	int raw_fd = (int)syscall(SYS_openat, AT_FDCWD, first->c_str(),
				  O_RDONLY | O_CLOEXEC);
	REQUIRE(raw_fd >= 0);
	char raw_buffer[32] = {};
	REQUIRE(syscall(SYS_read, raw_fd, raw_buffer, sizeof(raw_buffer)) ==
		(long)expected.size());
	REQUIRE(expected == raw_buffer);
	REQUIRE(syscall(SYS_close, raw_fd) == 0);

	// Metadata syscalls succeed on the redirected host path, including
	// the exact operations rootless loaders issue before opening.
	struct stat at_st = {};
	REQUIRE(fstatat(AT_FDCWD, first->c_str(), &at_st, 0) == 0);
	REQUIRE(S_ISREG(at_st.st_mode));
	struct statx stx = {};
	REQUIRE(syscall(SYS_statx, AT_FDCWD, first->c_str(), 0, STATX_ALL,
			&stx) == 0);
	REQUIRE((stx.stx_mask & STATX_MODE) != 0);
	REQUIRE(S_ISREG(stx.stx_mode));
	REQUIRE(faccessat(AT_FDCWD, first->c_str(), R_OK, 0) == 0);

	// A second virtual path reuses the same private directory instead of
	// allocating per-open temp files.
	auto kprobe = bpftime::resolve_redirected_path(
		KPROBE_TYPE_VIRTUAL_PATH);
	REQUIRE(kprobe.has_value());
	REQUIRE(std::filesystem::path(*kprobe).parent_path() ==
		synthetic_dir);
	size_t entries_after = 0;
	for (const auto &entry :
	     std::filesystem::directory_iterator(synthetic_dir))
		++entries_after;
	REQUIRE(entries_after == 2);
}

TEST_CASE("explicit BTF and kallsyms overrides redirect without copies",
	  "[syscall-server][mock-file][synthetic-fs]")
{
	char btf_template[] = "/tmp/bpftime-btf-override-XXXXXX";
	int btf_fd = mkstemp(btf_template);
	REQUIRE(btf_fd >= 0);
	REQUIRE(write(btf_fd, "BTF", 3) == 3);
	REQUIRE(close(btf_fd) == 0);
	BOOST_SCOPE_EXIT_ALL(&) { unlink(btf_template); };
	setenv("BPFTIME_BTF_PATH", btf_template, 1);
	BOOST_SCOPE_EXIT_ALL(&) { unsetenv("BPFTIME_BTF_PATH"); };
	auto btf = bpftime::resolve_redirected_path(
		"/sys/kernel/btf/vmlinux");
	REQUIRE(btf.has_value());
	REQUIRE(*btf == btf_template);
	// An empty value means "unset": no redirect, no generated copy.
	setenv("BPFTIME_BTF_PATH", "", 1);
	REQUIRE_FALSE(bpftime::resolve_redirected_path(
		"/sys/kernel/btf/vmlinux").has_value());

	char kallsyms_template[] = "/tmp/bpftime-kallsyms-override-XXXXXX";
	int kallsyms_fd = mkstemp(kallsyms_template);
	REQUIRE(kallsyms_fd >= 0);
	REQUIRE(write(kallsyms_fd, "0 T _stext\n", 11) == 11);
	REQUIRE(close(kallsyms_fd) == 0);
	BOOST_SCOPE_EXIT_ALL(&) { unlink(kallsyms_template); };
	setenv("BPFTIME_KALLSYMS_PATH", kallsyms_template, 1);
	BOOST_SCOPE_EXIT_ALL(&) { unsetenv("BPFTIME_KALLSYMS_PATH"); };
	auto kallsyms = bpftime::resolve_redirected_path("/proc/kallsyms");
	REQUIRE(kallsyms.has_value());
	REQUIRE(*kallsyms == kallsyms_template);
	setenv("BPFTIME_KALLSYMS_PATH", "", 1);
	auto without_override =
		bpftime::resolve_redirected_path("/proc/kallsyms");
	if (without_override)
		REQUIRE(*without_override != kallsyms_template);
}

TEST_CASE("synthetic fs private directory is removed at normal exit",
	  "[syscall-server][mock-file][synthetic-fs]")
{
	int pipefd[2];
	REQUIRE(pipe(pipefd) == 0);
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		close(pipefd[0]);
		auto resolved = bpftime::resolve_redirected_path(
			UPROBE_TYPE_VIRTUAL_PATH);
		if (!resolved)
			_exit(2);
		auto dir = std::filesystem::path(*resolved).parent_path();
		ssize_t written =
			write(pipefd[1], dir.c_str(), dir.string().size());
		(void)written;
		close(pipefd[1]);
		// exit() runs the atexit cleanup; _exit() would not.
		exit(0);
	}
	close(pipefd[1]);
	char buffer[512] = {};
	size_t total = 0;
	for (;;) {
		ssize_t n = read(pipefd[0], buffer + total,
				 sizeof(buffer) - 1 - total);
		if (n <= 0)
			break;
		total += static_cast<size_t>(n);
	}
	close(pipefd[0]);
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	REQUIRE(total > 0);
	std::error_code ec;
	REQUIRE_FALSE(
		std::filesystem::exists(std::filesystem::path(buffer), ec));
}
#endif
