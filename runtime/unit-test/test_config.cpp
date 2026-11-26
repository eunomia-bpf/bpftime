#include <catch2/catch_test_macros.hpp>
#include <bpftime_config.hpp>
#include "bpftime_shm_internal.hpp"
#include "./common_def.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
using namespace bpftime;
using namespace boost::interprocess;
static const char *SHM_NAME = "_BPFTIME_CONFIG_TEST";

static std::string test_string = "aaaabbb";

static std::string test_string_2 = "aaaassssbbb";

TEST_CASE("Test bpftime runtime_config")
{
	::shm_remove remover(SHM_NAME);
	managed_shared_memory mem(create_only, SHM_NAME, 20 << 20);

	runtime_config cfg;
	cfg.set_logger_output_path(test_string.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string);
	cfg.set_logger_output_path(test_string_2.c_str());
	REQUIRE(cfg.get_logger_output_path() == test_string_2);
}

TEST_CASE("Runtime config is available without a shared config")
{
	bpftime_shm shm(SHM_NAME, shm_open_type::SHM_NO_CREATE);
	REQUIRE(std::string(shm.get_runtime_config().get_vm_name()) ==
		construct_runtime_config_from_env().get_vm_name());
}

TEST_CASE("Default VM matches an enabled backend")
{
	runtime_config cfg;
#if defined(BPFTIME_LLVM_JIT)
	REQUIRE(std::string(cfg.get_vm_name()) == "llvm");
#elif defined(BPFTIME_UBPF_JIT)
	REQUIRE(std::string(cfg.get_vm_name()) == "ubpf");
#else
	REQUIRE(std::string(cfg.get_vm_name()).empty());
#endif
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

TEST_CASE("Select Frida's fuzzy backtracer from the environment")
{
	const char *old_value = getenv("BPFTIME_FRIDA_FUZZY_BACKTRACER");
	const bool was_set = old_value != nullptr;
	const std::string saved_value = was_set ? old_value : "";

	REQUIRE(unsetenv("BPFTIME_FRIDA_FUZZY_BACKTRACER") == 0);
	REQUIRE_FALSE(construct_runtime_config_from_env()
			      .enable_frida_fuzzy_backtracer);
	REQUIRE(setenv("BPFTIME_FRIDA_FUZZY_BACKTRACER", "1", 1) == 0);
	REQUIRE(construct_runtime_config_from_env()
			.enable_frida_fuzzy_backtracer);

	const int restore_result =
		was_set ? setenv("BPFTIME_FRIDA_FUZZY_BACKTRACER",
				 saved_value.c_str(), 1) :
			  unsetenv("BPFTIME_FRIDA_FUZZY_BACKTRACER");
	REQUIRE(restore_result == 0);
}

namespace
{
struct config_env_guard {
	const char *name;
	bool had;
	std::string previous;
	explicit config_env_guard(const char *key)
		: name(key), had(getenv(key) != nullptr),
		  previous(had ? getenv(key) : "")
	{
	}
	~config_env_guard()
	{
		had ? setenv(name, previous.c_str(), 1) : unsetenv(name);
		reload_agent_bootstrap_config_from_env();
	}
};
} // namespace

TEST_CASE("Bootstrap captures once until explicit reload and preserves errno",
	  "[config]")
{
	config_env_guard guard("BPFTIME_AGENT_ALLOW_PROCESS");
	REQUIRE(setenv(guard.name, "victim", 1) == 0);
	errno = EBUSY;
	reload_agent_bootstrap_config_from_env();
	REQUIRE(errno == EBUSY);
	REQUIRE(setenv(guard.name, "changed", 1) == 0);
	errno = EDOM;
	REQUIRE(std::string(get_agent_bootstrap_config().allow_process) ==
		"victim");
	REQUIRE(errno == EDOM);
	reload_agent_bootstrap_config_from_env();
	REQUIRE(errno == EDOM);
	REQUIRE(std::string(get_agent_bootstrap_config().allow_process) ==
		"changed");
}

TEST_CASE(
	"Local runtime mirror survives environment mutation and follows config updates",
	"[config]")
{
	config_env_guard guard("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	initialize_local_runtime_config_from_env();
	const runtime_config previous = bpftime_get_local_runtime_config();
	REQUIRE(setenv(guard.name, "8192", 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(shm_open_type::SHM_NO_CREATE);
	REQUIRE(bpftime_get_local_runtime_config()
			.system_analysis_ptregs_offset == 8192);
	REQUIRE(setenv(guard.name, "0", 1) == 0);
	REQUIRE(bpftime_get_local_runtime_config()
			.system_analysis_ptregs_offset == 8192);
	auto cfg = bpftime_get_runtime_config();
	cfg.system_analysis_ptregs_offset = 16384;
	bpftime_set_runtime_config(std::move(cfg));
	REQUIRE(bpftime_get_local_runtime_config()
			.system_analysis_ptregs_offset == 16384);
	::shm_remove remover(SHM_NAME);
	bpftime_shm owner(SHM_NAME, shm_open_type::SHM_REMOVE_AND_CREATE);
	bpftime_set_local_runtime_config(previous);
	bpftime_shm reader(SHM_NAME, shm_open_type::SHM_OPEN_ONLY);
	REQUIRE(bpftime_get_local_runtime_config()
			.system_analysis_ptregs_offset == 16384);
	bpftime_set_runtime_config(runtime_config(previous));
}

TEST_CASE("Numeric configuration preserves source type limits before clamping",
	  "[config]")
{
	config_env_guard memory("BPFTIME_SHM_MEMORY_MB");
	config_env_guard device("BPFTIME_CUDA_DEVICE");
	REQUIRE(setenv(memory.name, "2147483648", 1) == 0);
	REQUIRE(setenv(device.name, "2147483648", 1) == 0);
	const auto cfg = construct_runtime_config_from_env();
	REQUIRE(cfg.shm_memory_size == 50);
	REQUIRE(cfg.cuda_device == 0);
	REQUIRE(setenv(memory.name, "0", 1) == 0);
	REQUIRE(construct_runtime_config_from_env().shm_memory_size == 1);
}
