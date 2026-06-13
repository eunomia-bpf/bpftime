#include "catch2/catch_test_macros.hpp"
#include "nv_gpu_device_manager.hpp"
#include <cuda.h>
#include <spdlog/spdlog.h>

using namespace bpftime;
using namespace attach;

TEST_CASE("gpu_device_manager initialization")
{
	// cuInit is required before using the driver API
	cuInit(0);

	gpu_device_manager manager;
	manager.initialize();

	SECTION("device_count should be non-negative")
	{
		REQUIRE(manager.device_count() >= 0);
	}

	SECTION("devices vector size matches device_count")
	{
		REQUIRE((int)manager.devices().size() == manager.device_count());
	}

	if (manager.device_count() > 0) {
		SECTION("device 0 should have valid SM arch")
		{
			auto &dev = manager.get_default_device();
			REQUIRE(dev.device_ordinal == 0);
			REQUIRE(dev.sm_arch.substr(0, 3) == "sm_");
			REQUIRE(dev.module_pool != nullptr);
		}

		SECTION("get_unique_sm_archs should be non-empty")
		{
			auto archs = manager.get_unique_sm_archs();
			REQUIRE(!archs.empty());
		}

		SECTION("get_current_device should return a valid device")
		{
			auto &dev = manager.get_current_device();
			REQUIRE(dev.device_ordinal >= 0);
			REQUIRE(dev.device_ordinal < manager.device_count());
		}

		SECTION("out of range ordinal should throw")
		{
			REQUIRE_THROWS_AS(
				manager.get_device(manager.device_count()),
				std::out_of_range);
			REQUIRE_THROWS_AS(manager.get_device(-1),
					  std::out_of_range);
		}

		if (manager.device_count() > 1) {
			SECTION("multi-GPU: each device should have valid info")
			{
				for (int i = 0; i < manager.device_count();
				     i++) {
					auto &dev = manager.get_device(i);
					REQUIRE(dev.device_ordinal == i);
					REQUIRE(dev.sm_arch.substr(0, 3) ==
						"sm_");
					REQUIRE(dev.module_pool != nullptr);
					SPDLOG_INFO("Device {}: sm_arch={}",
						    i, dev.sm_arch);
				}
			}

			SECTION("multi-GPU: per-device module pools are "
				"separate")
			{
				auto &dev0 = manager.get_device(0);
				auto &dev1 = manager.get_device(1);
				REQUIRE(dev0.module_pool != dev1.module_pool);
			}
		}
	}
}

TEST_CASE("gpu_device_manager with BPFTIME_SM_ARCH override")
{
	// Set env var to override
	setenv("BPFTIME_SM_ARCH", "sm_99", 1);

	gpu_device_manager manager;
	manager.initialize();

	if (manager.device_count() > 0) {
		for (int i = 0; i < manager.device_count(); i++) {
			REQUIRE(manager.get_device(i).sm_arch == "sm_99");
		}
	}

	// Clean up
	unsetenv("BPFTIME_SM_ARCH");
}
