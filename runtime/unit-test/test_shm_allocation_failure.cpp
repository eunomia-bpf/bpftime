/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "shm_allocation_test_paths.hpp"

#include "bpftime_shm_internal.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace
{
struct helper_result {
	int status;
	std::string output;
};

struct env_var_guard {
	const char *name;
	std::string old_value;
	bool had_value;

	explicit env_var_guard(const char *env_name)
		: name(env_name), old_value(), had_value(getenv(env_name) != nullptr)
	{
		if (had_value) {
			old_value = getenv(env_name);
		}
	}

	~env_var_guard()
	{
		if (had_value) {
			setenv(name, old_value.c_str(), 1);
		} else {
			unsetenv(name);
		}
	}
};

struct parent_shm_guard {
	bool initialized = false;
	bool pid_added = false;
	bool added_as_syscall_server = false;

	void cleanup()
	{
		if (!initialized) {
			return;
		}
		if (pid_added) {
			if (added_as_syscall_server) {
				bpftime::shm_holder.global_shared_memory
					.remove_pid_from_alive_syscall_server_set(
						getpid());
			} else {
				bpftime::shm_holder.global_shared_memory
					.remove_pid_from_alive_agent_set(
						getpid());
			}
			pid_added = false;
		}
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_NO_CREATE);
		initialized = false;
	}

	~parent_shm_guard()
	{
		cleanup();
	}
};

enum class live_pid_set { alive_agent, syscall_server };

helper_result run_allocation_helper(const char *mode, const char *memory_mb,
				    const char *max_fd_count,
				    const char *log_output,
				    const char *log_level = nullptr,
				    bool cleanup_shm = true,
				    std::string *shm_name_out = nullptr,
				    bool remove_before_start = true)
{
	const std::string shm_name = "bpftime-shm-allocation-test-" +
				     std::string(mode) + "-" +
				     std::to_string(getpid());
	if (shm_name_out != nullptr) {
		*shm_name_out = shm_name;
	}
	if (remove_before_start) {
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str());
	}
	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1) {
			_exit(125);
		}
		close(output_pipe[1]);
		if (unsetenv("BPFTIME_LOG_OUTPUT") != 0 ||
		    unsetenv("SPDLOG_LEVEL") != 0 ||
		    unsetenv("BPFTIME_SHM_MEMORY_MB") != 0 ||
		    unsetenv("BPFTIME_MAX_FD_COUNT") != 0 ||
		    setenv("HOME", "/proc/bpftime-unwritable-home", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0) {
			_exit(126);
		}
		if (memory_mb != nullptr &&
		    setenv("BPFTIME_SHM_MEMORY_MB", memory_mb, 1) != 0)
			_exit(126);
		if (max_fd_count != nullptr &&
		    setenv("BPFTIME_MAX_FD_COUNT", max_fd_count, 1) != 0)
			_exit(126);
		if (log_output != nullptr &&
		    setenv("BPFTIME_LOG_OUTPUT", log_output, 1) != 0)
			_exit(126);
		if (log_level != nullptr &&
		    setenv("SPDLOG_LEVEL", log_level, 1) != 0)
			_exit(126);
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, mode, nullptr);
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
	while (waitpid(pid, &status, 0) == -1) {
		REQUIRE(errno == EINTR);
	}
	if (cleanup_shm) {
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str());
	}
	return { status, std::move(output) };
}

bool wait_for_file(const std::string &path)
{
	for (int i = 0; i < 3000; i++) {
		if (access(path.c_str(), F_OK) == 0) {
			return true;
		}
		usleep(10000);
	}
	return false;
}

helper_result run_waiting_helper_with_live_pid(const std::string &shm_name,
					       live_pid_set set_name)
{
	const std::string ready_file =
		"/tmp/bpftime-shm-allocation-ready-" + std::to_string(getpid());
	const std::string release_file =
		"/tmp/bpftime-shm-allocation-release-" +
		std::to_string(getpid());
	unlink(ready_file.c_str());
	unlink(release_file.c_str());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());

	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);
	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1) {
			_exit(125);
		}
		close(output_pipe[1]);
		if (setenv("HOME", "/proc/bpftime-unwritable-home", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("BPFTIME_SHM_MEMORY_MB", "4", 1) != 0 ||
		    setenv("BPFTIME_MAX_FD_COUNT", "128", 1) != 0 ||
		    setenv("BPFTIME_HELPER_READY_FILE", ready_file.c_str(),
			   1) != 0 ||
		    setenv("BPFTIME_HELPER_RELEASE_FILE", release_file.c_str(),
			   1) != 0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0) {
			_exit(126);
		}
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, "startup-wait",
		      nullptr);
		_exit(127);
	}

	close(output_pipe[1]);
	REQUIRE(wait_for_file(ready_file));

	env_var_guard shm_name_guard("BPFTIME_GLOBAL_SHM_NAME");
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	parent_shm_guard parent_shm;
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_OPEN_ONLY);
	parent_shm.initialized = true;
	if (set_name == live_pid_set::syscall_server) {
		bpftime::shm_holder.global_shared_memory
			.add_pid_into_alive_syscall_server_set(getpid());
		parent_shm.added_as_syscall_server = true;
	} else {
		bpftime::shm_holder.global_shared_memory
			.add_pid_into_alive_agent_set(getpid());
	}
	parent_shm.pid_added = true;

	{
		std::ofstream release(release_file);
		REQUIRE(release.good());
	}

	std::string output;
	char buf[256];
	for (;;) {
		ssize_t bytes_read = read(output_pipe[0], buf, sizeof(buf));
		if (bytes_read < 0) {
			REQUIRE(errno == EINTR);
			continue;
		}
		if (bytes_read == 0) {
			break;
		}
		output.append(buf, buf + bytes_read);
	}
	close(output_pipe[0]);

	int status = 0;
	while (waitpid(pid, &status, 0) == -1) {
		REQUIRE(errno == EINTR);
	}
	parent_shm.cleanup();
	unlink(ready_file.c_str());
	unlink(release_file.c_str());
	return { status, std::move(output) };
}

helper_result run_agent_preload_without_shm()
{
	const std::string shm_name = "bpftime-agent-missing-shm-test-" +
				     std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1) {
			_exit(125);
		}
		close(output_pipe[1]);
		if (unsetenv("BPFTIME_LOG_OUTPUT") != 0 ||
		    unsetenv("SPDLOG_LEVEL") != 0 ||
		    setenv("HOME", "/proc/bpftime-unwritable-home", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("LD_PRELOAD", BPFTIME_AGENT_LIBRARY, 1) != 0) {
			_exit(126);
		}
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, "check-sigusr1",
		      nullptr);
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
	while (waitpid(pid, &status, 0) == -1) {
		REQUIRE(errno == EINTR);
	}
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	return { status, std::move(output) };
}

size_t count_occurrences(const std::string &text, const std::string &needle)
{
	size_t count = 0;
	size_t pos = 0;
	while ((pos = text.find(needle, pos)) != std::string::npos) {
		count++;
		pos += needle.size();
	}
	return count;
}
} // namespace

TEST_CASE("Syscall server falls back when startup shared memory is too small",
	  "[allocation][syscall_server]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", "console");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE_FALSE(result.output.empty());
	REQUIRE(count_occurrences(result.output, "Starting syscall server") == 1);
}

TEST_CASE("Syscall server perf mmap reports shared memory exhaustion",
	  "[allocation][syscall_server]")
{
	auto result = run_allocation_helper("perf-mmap", "4", "128", "console");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
}

TEST_CASE("Syscall server keeps logger sink failures off host stdio",
	  "[allocation][syscall_server][logging]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", "/dev/full");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("Syscall server keeps default logging off host stdio",
	  "[allocation][syscall_server][logging]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", nullptr);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("Global log level cannot enable default host stdio",
	  "[allocation][syscall_server][logging]")
{
	auto result = run_allocation_helper("startup", "64", "1048576", nullptr,
					    "info");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("File logger is installed before syscall startup",
	  "[allocation][syscall_server][logging]")
{
	const std::string log_path = "/tmp/bpftime-syscall-startup-" +
				     std::to_string(getpid()) + ".log";
	unlink(log_path.c_str());
	auto result = run_allocation_helper("perf-mmap", "4", "128",
					    log_path.c_str());
	std::ifstream log(log_path);
	std::string contents{ std::istreambuf_iterator<char>(log), {} };
	unlink(log_path.c_str());

	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(result.output.empty());
	REQUIRE(contents.find("Starting syscall server") != std::string::npos);
	REQUIRE(contents.find("bpftime-syscall-server started") !=
		std::string::npos);
}

TEST_CASE("Syscall server preserves the console logger level name",
	  "[allocation][syscall_server][logging]")
{
	auto result = run_allocation_helper("startup", "64", "1048576",
					    "console", "stderr=off");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("Agent preload keeps missing shared memory off host stdio",
	  "[allocation][agent][logging]")
{
	auto result = run_agent_preload_without_shm();
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("Syscall server removes owned shared memory on normal exit",
	  "[allocation][syscall_server][cleanup]")
{
	std::string shm_name;
	auto result = run_allocation_helper("startup-ok", "4", "128", nullptr,
					    nullptr, false, &shm_name);

	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE_FALSE(
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str()));
}

TEST_CASE("Syscall server preserves shared memory it only opened",
	  "[allocation][syscall_server][cleanup]")
{
	const std::string shm_name = "bpftime-shm-allocation-test-startup-ok-" +
				     std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	boost::interprocess::managed_shared_memory existing_segment(
		boost::interprocess::create_only, shm_name.c_str(), 4 << 20);

	std::string helper_shm_name;
	auto result = run_allocation_helper("startup-ok", "4", "128", nullptr,
					    nullptr, false, &helper_shm_name,
					    false);

	REQUIRE(helper_shm_name == shm_name);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str()));
}

TEST_CASE("Syscall server keeps shared memory while an agent pid is alive",
	  "[allocation][syscall_server][cleanup]")
{
	const std::string shm_name =
		"bpftime-shm-allocation-test-live-agent-" +
		std::to_string(getpid());
	auto result =
		run_waiting_helper_with_live_pid(shm_name,
						 live_pid_set::alive_agent);

	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(result.output.empty());
	REQUIRE(
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str()));
}

TEST_CASE("Syscall server keeps shared memory while a peer server pid is alive",
	  "[allocation][syscall_server][cleanup]")
{
	const std::string shm_name =
		"bpftime-shm-allocation-test-live-server-" +
		std::to_string(getpid());
	auto result =
		run_waiting_helper_with_live_pid(shm_name,
						 live_pid_set::syscall_server);

	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(result.output.empty());
	REQUIRE(
		boost::interprocess::shared_memory_object::remove(
			shm_name.c_str()));
}

TEST_CASE("Syscall server mock files do not leave tmp entries",
	  "[allocation][syscall_server][cleanup]")
{
	auto result = run_allocation_helper("mock-tmp", "4", "128", nullptr);

	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(result.output.empty());
}
#endif
