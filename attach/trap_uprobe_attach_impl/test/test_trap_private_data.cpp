#include "trap_test_common.hpp"
#include "trap_attach_private_data.hpp"
#include "trap_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <climits>
#include <cstdint>
#include <string>
#include <unistd.h>

using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET int __trap_private_data_target(int x)
{
	asm("");
	return x + 1;
}

static std::string self_exe()
{
	char buf[PATH_MAX] = {};
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	REQUIRE(n > 0);
	return std::string(buf, (size_t)n);
}

TEST_CASE("Trap backend: private data from a plain address")
{
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string("123456") == 0);
	REQUIRE(priv.addr == 123456);
	REQUIRE(priv.module_name.empty());
	REQUIRE(priv.to_string() == "123456");
}

TEST_CASE("Trap backend: private data from module and offset")
{
	auto exe = self_exe();
	void *base = get_module_base_addr("");
	REQUIRE(base != nullptr);
	REQUIRE(get_module_base_addr(exe.c_str()) == base);
	uintptr_t offset = (uintptr_t)&__trap_private_data_target -
			   (uintptr_t)base;
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string(exe + ":" +
					    std::to_string(offset)) == 0);
	REQUIRE(priv.addr == (uintptr_t)&__trap_private_data_target);
	REQUIRE(priv.module_name == exe);
}

TEST_CASE("Trap backend: private data rejects malformed input")
{
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string("/bin/ls:") == -EINVAL);
	REQUIRE(priv.initialize_from_string("not-a-number") == -EINVAL);
	REQUIRE(priv.initialize_from_string("/proc/1/map_files/0-0:1") ==
		-ENOENT);
}

TEST_CASE("Trap backend: map_files module names resolve to the mapped path")
{
	REQUIRE(resolve_mapped_module_path("/usr/lib/libc.so.6") ==
		std::string("/usr/lib/libc.so.6"));
	// A range belonging to another process cannot be resolved
	REQUIRE(!resolve_mapped_module_path("/proc/1/map_files/1000-2000"));
}
