#include <catch2/catch_test_macros.hpp>
#include <bpftime_config.hpp>
#include "./common_def.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
using namespace bpftime;
using namespace boost::interprocess;
static const char *SHM_NAME = "_BPFTIME_CONFIG_TEST";

static std::string test_string = "aaaabbb";

static std::string test_string_2 = "aaaassssbbb";

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
	const auto cfg = construct_agent_config_from_env();
	const int restore_result =
		was_set ? setenv("BPFTIME_ALLOW_EXTERNAL_MAPS",
				 saved_value.c_str(), 1) :
			  unsetenv("BPFTIME_ALLOW_EXTERNAL_MAPS");

	REQUIRE(restore_result == 0);
	REQUIRE(cfg.allow_non_buildin_map_types);
}
