#include <catch2/catch_test_macros.hpp>
#include <bpftime_config.hpp>
#include <cerrno>
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
	REQUIRE(cfg.probe_read_window_bytes == 512);
	REQUIRE(cfg.probe_read_lookbehind_bytes == 64);
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

TEST_CASE("Probe window environment clamps size and lookbehind")
{
	for (auto name : { "BPFTIME_PROBE_READ_WINDOW_BYTES", "BPFTIME_PROBE_READ_LOOKBEHIND_BYTES" }) {
		const char *old = getenv(name);
		const std::string saved = old ? old : "";
		setenv(name, "8192", 1);
		const auto cfg = construct_runtime_config_from_env();
		if (old) setenv(name, saved.c_str(), 1); else unsetenv(name);
		if (name == std::string("BPFTIME_PROBE_READ_WINDOW_BYTES"))
			REQUIRE(cfg.probe_read_window_bytes == 512);
		else REQUIRE(cfg.probe_read_lookbehind_bytes == cfg.probe_read_window_bytes);
	}
}

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

TEST_CASE("Startup POD snapshots survive environment changes", "[config]")
{
	static_assert(std::is_trivially_copyable_v<agent_bootstrap_config>);
	static_assert(std::is_trivially_copyable_v<runtime_config>);
	config_env_guard admission("BPFTIME_AGENT_ALLOW_PROCESS");
	config_env_guard threads("BPFTIME_MAP_GPU_THREAD_COUNT");
	config_env_guard procfs("BPFTIME_HOST_PROCFS");
	REQUIRE(setenv(admission.name, "victim", 1) == 0);
	REQUIRE(setenv(threads.name, "8", 1) == 0);
	REQUIRE(setenv(procfs.name, "/collector-procfs", 1) == 0);
	errno = EBUSY;
	reload_agent_bootstrap_config_from_env();
	REQUIRE(errno == EBUSY);
	::shm_remove remover(SHM_NAME);
	bpftime_shm owner(SHM_NAME, shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(setenv(admission.name, "changed", 1) == 0);
	REQUIRE(setenv(threads.name, "16", 1) == 0);
	errno = EDOM;
	REQUIRE(std::string(get_agent_bootstrap_config().allow_process) ==
		"victim");
	REQUIRE(owner.get_runtime_config().map_gpu_thread_count == 8);
	REQUIRE(owner.get_runtime_config().has_map_gpu_thread_count);
	REQUIRE(errno == EDOM);
	REQUIRE(setenv(procfs.name, "/target-procfs", 1) == 0);
	reload_agent_bootstrap_config_from_env();
	bpftime_shm reader(SHM_NAME, shm_open_type::SHM_OPEN_ONLY);
	REQUIRE(reader.get_runtime_config().map_gpu_thread_count == 8);
	REQUIRE(std::string(get_agent_bootstrap_config().host_procfs) ==
		"/target-procfs");
}
