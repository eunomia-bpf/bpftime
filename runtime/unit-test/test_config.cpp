#include <catch2/catch_test_macros.hpp>
#include <bpftime_config.hpp>
#include "./common_def.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
using namespace bpftime;
using namespace boost::interprocess;
static const char *SHM_NAME = "_BPFTIME_CONFIG_TEST";

static std::string test_string = "aaaabbb";

static std::string test_string_2 = "aaaassssbbb";

TEST_CASE("Test bpftime agent_config")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);

	agent_config cfg;
	cfg.set_logger_output_path(test_string.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string);
	cfg.change_to_shm_object(mem);
	REQUIRE(cfg.get_logger_output_path() == test_string);
	cfg.set_logger_output_path(test_string_2.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string_2);
}
