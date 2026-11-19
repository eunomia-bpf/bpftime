/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * Map semantics observed from a real caller of the intercepted bpf()
 * syscall, which the in-process tests cannot cover: the shm-level wrappers
 * take a different path from the helper entry points, and the errno the
 * syscall layer reports is what libbpf-based tools see.
 */
#include "catch2/catch_test_macros.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#if __linux__
#include "shm_allocation_test_paths.hpp"

namespace
{
struct child_result {
	int status = -1;
	std::map<std::string, std::string> fields;
};

// Parse `key=value` lines from the helper's stdout into a field map, so a
// failure names the observed value rather than only the exit status.
std::map<std::string, std::string> parse_fields(const std::string &output)
{
	std::map<std::string, std::string> fields;
	size_t start = 0;
	while (start < output.size()) {
		size_t end = output.find('\n', start);
		if (end == std::string::npos)
			end = output.size();
		const std::string line = output.substr(start, end - start);
		const size_t equals = line.find('=');
		if (equals != std::string::npos)
			fields[line.substr(0, equals)] = line.substr(equals + 1);
		start = end + 1;
	}
	return fields;
}

child_result run_map_helper(const char *mode, const char *shm_name)
{
	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);
	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1)
			_exit(125);
		close(output_pipe[1]);
		if (setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name, 1) != 0 ||
		    setenv("BPFTIME_SHM_MEMORY_MB", "16", 1) != 0 ||
		    setenv("BPFTIME_LOG_OUTPUT", "/dev/null", 1) != 0 ||
		    unsetenv("SPDLOG_LEVEL") != 0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0)
			_exit(126);
		execl(BPFTIME_MAP_SYSCALL_TEST_HELPER,
		      BPFTIME_MAP_SYSCALL_TEST_HELPER, mode, nullptr);
		_exit(127);
	}
	close(output_pipe[1]);
	std::string output;
	char buffer[4096];
	for (;;) {
		ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
		if (count > 0) {
			output.append(buffer, static_cast<size_t>(count));
			continue;
		}
		if (count == -1 && errno == EINTR)
			continue;
		REQUIRE(count == 0);
		break;
	}
	close(output_pipe[0]);
	int status = 0;
	while (waitpid(pid, &status, 0) == -1)
		REQUIRE(errno == EINTR);

	child_result result;
	result.status = status;
	result.fields = parse_fields(output);
	return result;
}

std::string shm_name_for(const char *suffix)
{
	return std::string("bpftime-map-syscall-test-") + suffix + "-" +
	       std::to_string(getpid());
}

// Read a field as an integer, defaulting to a sentinel so a missing line
// fails the assertion instead of comparing garbage.
long field(const std::map<std::string, std::string> &fields, const char *name)
{
	auto itr = fields.find(name);
	if (itr == fields.end())
		return -1;
	return strtol(itr->second.c_str(), nullptr, 0);
}

unsigned long long ufield(const std::map<std::string, std::string> &fields,
			  const char *name)
{
	auto itr = fields.find(name);
	if (itr == fields.end())
		return ~0ULL;
	return strtoull(itr->second.c_str(), nullptr, 0);
}

} // namespace

TEST_CASE("intercepted lookup preserves the backend errno", "[map_syscall]")
{
	const std::string shm_name = shm_name_for("lookup-errno");
	remove(("/dev/shm/" + shm_name).c_str());
	auto result = run_map_helper("lookup-errno", shm_name.c_str());
	remove(("/dev/shm/" + shm_name).c_str());

	REQUIRE(result.status == 0);
	// A write-only map rejects an fd-based read with EPERM; the syscall
	// layer must not replace that with ENOENT.
	REQUIRE(field(result.fields, "lookup_ret") == -1);
	REQUIRE(field(result.fields, "lookup_errno") == EPERM);
	// An absent key is still ENOENT.
	REQUIRE(field(result.fields, "missing_ret") == -1);
	REQUIRE(field(result.fields, "missing_errno") == ENOENT);
}

TEST_CASE("intercepted lookup-and-delete is keyed for hash maps",
	  "[map_syscall]")
{
	const std::string shm_name = shm_name_for("lookup-and-delete");
	remove(("/dev/shm/" + shm_name).c_str());
	auto result = run_map_helper("lookup-and-delete", shm_name.c_str());
	remove(("/dev/shm/" + shm_name).c_str());

	REQUIRE(result.status == 0);
	// The value comes back and the entry is removed.
	REQUIRE(field(result.fields, "hash_ret") == 0);
	REQUIRE(ufield(result.fields, "hash_value") ==
		0xaabbccddeeff0011ULL);
	REQUIRE(field(result.fields, "after_ret") == -1);
	REQUIRE(field(result.fields, "after_errno") == ENOENT);
	REQUIRE(field(result.fields, "again_ret") == -1);
	REQUIRE(field(result.fields, "again_errno") == ENOENT);
}

TEST_CASE("intercepted lookup-and-delete keeps the queue pop path",
	  "[map_syscall]")
{
	const std::string shm_name = shm_name_for("queue-pop");
	remove(("/dev/shm/" + shm_name).c_str());
	auto result = run_map_helper("queue-pop", shm_name.c_str());
	remove(("/dev/shm/" + shm_name).c_str());

	REQUIRE(result.status == 0);
	REQUIRE(field(result.fields, "queue_ret") == 0);
	REQUIRE(ufield(result.fields, "queue_value") ==
		0x0123456789abcdefULL);
	REQUIRE(field(result.fields, "queue_empty_ret") == -1);
	REQUIRE(field(result.fields, "queue_empty_errno") == ENOENT);
}
#endif