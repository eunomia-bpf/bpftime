#include <catch2/catch_test_macros.hpp>
#include <bpftime_config.hpp>
#include <bpftime_logger.hpp>
#include "./common_def.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>

#if defined(__linux__) && defined(BPFTIME_TEST_VERIFIER_DISABLED)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#include <string_view>
#include <utility>
#endif

using namespace bpftime;
using namespace boost::interprocess;
static const char *SHM_NAME = "_BPFTIME_CONFIG_TEST";

static std::string test_string = "aaaabbb";

static std::string test_string_2 = "aaaassssbbb";

#if defined(__linux__) && defined(BPFTIME_TEST_VERIFIER_DISABLED)
namespace
{
struct verifier_config_result {
	int status;
	std::string output;
};

verifier_config_result
run_verifier_config_helper(const char *level,
			   bpftime_verifier_mode expected_mode)
{
	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);

	const pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1) {
			_exit(125);
		}
		close(output_pipe[1]);
		bpftime_set_logger("console");
		const int env_result =
			level == nullptr ?
				unsetenv("BPFTIME_VERIFIER_LEVEL") :
				setenv("BPFTIME_VERIFIER_LEVEL", level, 1);
		if (env_result != 0)
			_exit(126);
		const auto config = construct_runtime_config_from_env();
		if (config.verifier_mode != expected_mode)
			_exit(127);
		bpftime_logger_flush();
		_exit(0);
	}

	close(output_pipe[1]);
	std::string output;
	char buffer[4096];
	for (;;) {
		const ssize_t count =
			read(output_pipe[0], buffer, sizeof(buffer));
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
	return { status, std::move(output) };
}
} // namespace
#endif

TEST_CASE("Test bpftime runtime_config")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);

	runtime_config cfg;
	cfg.set_logger_output_path(test_string.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string);
	cfg.set_logger_output_path(test_string_2.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string_2);
}

TEST_CASE("Allow external maps from the environment")
{
	const char *old_value = getenv("BPFTIME_ALLOW_EXTERNAL_MAPS");
	const bool was_set = old_value != nullptr;
	const std::string saved_value = was_set ? old_value : "";

	REQUIRE(setenv("BPFTIME_ALLOW_EXTERNAL_MAPS", "1", 1) == 0);
	const auto cfg = construct_runtime_config_from_env();
	const int restore_result =
		was_set ? setenv("BPFTIME_ALLOW_EXTERNAL_MAPS",
				 saved_value.c_str(), 1) :
			  unsetenv("BPFTIME_ALLOW_EXTERNAL_MAPS");

	REQUIRE(restore_result == 0);
	REQUIRE(cfg.allow_non_buildin_map_types);
}

#if defined(__linux__) && defined(BPFTIME_TEST_VERIFIER_DISABLED)
TEST_CASE("Verifier-disabled builds preserve non-strict modes",
	  "[config][verifier]")
{
	struct test_case {
		const char *level;
		bpftime_verifier_mode expected_mode;
	};
	const test_case cases[] = {
		{ nullptr, BPFTIME_VERIFIER_WARNING },
		{ "NO_VERIFY", BPFTIME_NO_VERIFY },
		{ "WARNING", BPFTIME_VERIFIER_WARNING },
	};
	for (const auto &test : cases) {
		const auto result = run_verifier_config_helper(
			test.level, test.expected_mode);
		REQUIRE(WIFEXITED(result.status));
		REQUIRE(WEXITSTATUS(result.status) == 0);
		if (test.level != nullptr &&
		    std::string_view(test.level) == "WARNING") {
			REQUIRE(result.output.find(
					"continuing without userspace verification") !=
				std::string::npos);
		}
	}
}

TEST_CASE("Verifier-disabled builds reject strict verification",
	  "[config][verifier]")
{
	const auto result =
		run_verifier_config_helper("STRICT", BPFTIME_VERIFIER_STRICT);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == EXIT_FAILURE);
	REQUIRE(result.output.find("requires userspace verifier support") !=
		std::string::npos);
}
#endif
