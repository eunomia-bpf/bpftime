#include <catch2/catch_test_macros.hpp>
#include "../agent/process_filter.hpp"
#include <cstring>
#include <string>
#if defined(__linux__)
#include "shm_allocation_test_paths.hpp"
#include <cerrno>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
using bpftime::agent::process_filter_skips; using bpftime::agent::process_name_in_list;
TEST_CASE("process filter defaults to allowing every process",
	  "[agent][process_filter]")
{
	REQUIRE_FALSE(process_filter_skips(nullptr, nullptr, "sh", "dash"));
	REQUIRE_FALSE(process_filter_skips("", "", "sh", "dash"));
	REQUIRE_FALSE(process_filter_skips(nullptr, "", nullptr, nullptr));
}
TEST_CASE("deny list matches a basename and wins over allow",
	  "[agent][process_filter]")
{
	REQUIRE(process_filter_skips("sh,bash", nullptr, "sh", "dash"));
	REQUIRE_FALSE(process_filter_skips("sh,bash", nullptr, "cat", "cat"));
	REQUIRE(process_filter_skips("bash", "bash", "bash", "bash"));
}
TEST_CASE("non-empty allow list requires a basename match",
	  "[agent][process_filter]")
{
	REQUIRE_FALSE(process_filter_skips(nullptr, "python3,python3.11",
					   "python3", "python3.11"));
	REQUIRE(process_filter_skips(nullptr, "python3", "sed", "sed"));
}
TEST_CASE("list tokens trim whitespace and empty tokens are ignored",
	  "[agent][process_filter]")
{
	const char *list = " python3 ,\tpython3.11, ,";
	REQUIRE(process_name_in_list("python3", list));
	REQUIRE(process_name_in_list("python3.11", list));
	REQUIRE_FALSE(process_name_in_list("python", list));
	REQUIRE_FALSE(process_name_in_list("python3", ""));
	REQUIRE_FALSE(process_name_in_list(nullptr, list));
}
TEST_CASE("invoked name or resolved exe name can match (symlinked exe)",
	  "[agent][process_filter]")
{
	REQUIRE_FALSE(process_filter_skips(nullptr, "python3",
					   "python3", "python3.11"));
	REQUIRE_FALSE(process_filter_skips(nullptr, "python3.11",
					   "python3", "python3.11"));
	REQUIRE(process_filter_skips(nullptr, "python3.10",
				    "python3", "python3.11"));
	REQUIRE(process_filter_skips("python3.11", "python3",
				    "python3", "python3.11"));
}
TEST_CASE("deleted executable marker is stripped before basename matching",
	  "[agent][process_filter]")
{
	using bpftime::agent::strip_deleted_marker;
	char deleted[] = "/usr/bin/python3.11 (deleted)";
	strip_deleted_marker(deleted);
	REQUIRE(strcmp(strrchr(deleted, '/') + 1, "python3.11") == 0);
	char plain[] = "/usr/bin/python3.11";
	strip_deleted_marker(plain);
	REQUIRE(strcmp(strrchr(plain, '/') + 1, "python3.11") == 0);
}
#if defined(__linux__)
// Snapshot/restore so a failure mid-case cannot leak admission env into the
// rest of the single-process test runner.
struct filter_env_guard {
	const char *name; bool had; std::string old;
	explicit filter_env_guard(const char *n)
		: name(n), had(getenv(n) != nullptr), old(had ? getenv(n) : "") {}
	~filter_env_guard()
	{
		had ? setenv(name, old.c_str(), 1) : unsetenv(name);
		bpftime::reload_agent_bootstrap_config_from_env();
	}
};
TEST_CASE("argv[0] override decides admission and errno is preserved",
	  "[agent][process_filter]")
{
	using bpftime::agent::should_skip_agent_process;
	filter_env_guard deny_guard("BPFTIME_AGENT_DENY_PROCESS");
	filter_env_guard allow_guard("BPFTIME_AGENT_ALLOW_PROCESS");
	setenv("BPFTIME_AGENT_DENY_PROCESS", "sh", 1);
	setenv("BPFTIME_AGENT_ALLOW_PROCESS", "", 1);
	bpftime::reload_agent_bootstrap_config_from_env();
	REQUIRE(should_skip_agent_process("sh"));
	REQUIRE_FALSE(should_skip_agent_process("python3"));
	setenv("BPFTIME_AGENT_DENY_PROCESS", "", 1);
	setenv("BPFTIME_AGENT_ALLOW_PROCESS", "python3", 1);
	bpftime::reload_agent_bootstrap_config_from_env();
	REQUIRE_FALSE(should_skip_agent_process("python3"));
	REQUIRE(should_skip_agent_process("sed"));
	errno = 42; (void)should_skip_agent_process("sed"); REQUIRE(errno == 42); errno = 0;
}
// The bootstrap logger opens its file sink eagerly, so the log file's
// existence is an init observable independent of SPDLOG_ACTIVE_LEVEL:
// a skipped process never installs the logger and never creates the file.
TEST_CASE("filtered process skips agent init under real LD_PRELOAD",
	  "[agent][process_filter]")
{
	const std::string shm = "bpftime-process-filter-test-" +
				std::to_string(getpid());
	auto run_shell = [&](const char *filter_var, const char *filter_value,
			     const std::string &log_path) {
		int pipefd[2];
		REQUIRE(pipe(pipefd) == 0);
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0) {
			close(pipefd[0]);
			dup2(pipefd[1], STDOUT_FILENO);
			dup2(pipefd[1], STDERR_FILENO);
			close(pipefd[1]);
			setenv("HOME", "/proc/bpftime-unwritable-home", 1);
			setenv("BPFTIME_GLOBAL_SHM_NAME", shm.c_str(), 1);
			setenv("BPFTIME_LOG_OUTPUT", log_path.c_str(), 1);
			setenv("LD_PRELOAD", BPFTIME_AGENT_LIBRARY, 1);
			unsetenv("BPFTIME_AGENT_DENY_PROCESS");
			unsetenv("BPFTIME_AGENT_ALLOW_PROCESS");
			if (filter_var)
				setenv(filter_var, filter_value, 1);
			execl("/bin/sh", "sh", "-c",
			      "echo process-filter-e2e-ok", nullptr);
			_exit(127);
		}
		close(pipefd[1]);
		std::string output;
		char buf[4096];
		for (;;) {
			ssize_t n = read(pipefd[0], buf, sizeof(buf));
			if (n > 0) { output.append(buf, static_cast<size_t>(n)); continue; }
			if (n < 0 && errno == EINTR) continue;
			break;
		}
		close(pipefd[0]);
		int status = 0;
		waitpid(pid, &status, 0);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
		REQUIRE(output.find("process-filter-e2e-ok") != std::string::npos);
		bool logger_installed = access(log_path.c_str(), F_OK) == 0;
		unlink(log_path.c_str());
		return logger_installed;
	};
	const std::string base = "/tmp/bpftime-process-filter-" +
				 std::to_string(getpid());
	// Negative control: without a filter the agent installs its logger.
	REQUIRE(run_shell(nullptr, nullptr, base + "-control.log"));
	// Filtered runs bypass logger/SHM entirely while still running sh.
	REQUIRE_FALSE(run_shell("BPFTIME_AGENT_DENY_PROCESS", "sh",
				base + "-deny.log"));
	REQUIRE_FALSE(run_shell("BPFTIME_AGENT_ALLOW_PROCESS",
				"definitely-not-a-shell", base + "-allow.log"));
}
// The text-transformer entrypoint must honor admission before swapping the
// syscall hook (and before any logging, since the bootstrap logger never ran).
TEST_CASE("syscall trace setup keeps the original hook when admission skips",
	  "[agent][process_filter]")
{
	using hooker_fn = int64_t (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
	filter_env_guard deny_guard("BPFTIME_AGENT_DENY_PROCESS");
	filter_env_guard allow_guard("BPFTIME_AGENT_ALLOW_PROCESS");
	char exe_buf[4096];
	ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
	REQUIRE(n > 0);
	exe_buf[n] = '\0';
	const char *slash = strrchr(exe_buf, '/');
	std::string self_name = slash != nullptr ? slash + 1 : exe_buf;
	setenv("BPFTIME_AGENT_DENY_PROCESS", self_name.c_str(), 1);
	unsetenv("BPFTIME_AGENT_ALLOW_PROCESS");
	void *handle = dlopen(BPFTIME_AGENT_LIBRARY, RTLD_NOW | RTLD_LOCAL);
	REQUIRE(handle != nullptr);
	auto setup = reinterpret_cast<void (*)(hooker_fn *)>(
		dlsym(handle, "_bpftime__setup_syscall_trace_callback"));
	REQUIRE(setup != nullptr);
	hooker_fn original = [](int64_t, auto...) -> int64_t { return 0; };
	hooker_fn hooker = original;
	setup(&hooker);
	REQUIRE(hooker == original);
	dlclose(handle);
}
#endif
