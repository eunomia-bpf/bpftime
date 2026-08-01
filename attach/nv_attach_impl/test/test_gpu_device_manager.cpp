#include "catch2/catch_test_macros.hpp"
#include "nv_attach_impl.hpp"
#include <cuda.h>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <stdexcept>

using namespace bpftime;
using namespace attach;

TEST_CASE("gpu_device_manager initialization")
{
	// cuInit is required before using the driver API
	cuInit(0);

	gpu_device_manager manager;
	manager.initialize();
	manager.initialize();

	SECTION("device_count should be non-negative")
	{
		REQUIRE(manager.device_count() >= 0);
	}

	if (manager.device_count() > 0) {
		SECTION("device 0 should have valid SM arch")
		{
			auto &dev = manager.get_device(0);
			REQUIRE(dev.device_ordinal == 0);
			REQUIRE(dev.sm_arch.substr(0, 3) == "sm_");
			REQUIRE(dev.module_pool != nullptr);
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
					SPDLOG_INFO("Device {}: sm_arch={}", i,
						    dev.sm_arch);
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

TEST_CASE("PTX compiler failures propagate to the caller")
{
	nv_attach_impl impl;
	impl.shared_mem_ptr = 1;
	impl.ptx_compiler.create = []() -> nv_attach_impl_ptx_compiler * {
		return nullptr;
	};

	fatbin_record record;
	record.module_pool = impl.module_pool;
	record.ptx_pool = impl.ptx_pool;
	record.original_ptx.emplace(
		"test.ptx", ".version 5.0\n.target sm_60\n.address_size 64\n");

	REQUIRE_THROWS_AS(record.try_loading_ptxs_for_device(impl, 0, "sm_60"),
			  std::runtime_error);
}
