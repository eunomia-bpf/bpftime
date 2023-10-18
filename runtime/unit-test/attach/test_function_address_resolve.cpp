#include "catch2/catch_message.hpp"
#include <catch2/catch_test_macros.hpp>
#include <attach/attach_manager/frida_attach_manager.hpp>
#include <cstdlib>
using namespace bpftime;

extern "C" int __func_reolve_test(int a, int b)
{
	return a + b;
}

TEST_CASE("Test internal function resolve")
{
	frida_attach_manager man;
	void *addr = man.find_function_addr_by_name("__func_reolve_test");
	REQUIRE(addr != nullptr);
	REQUIRE(addr == (void *)&__func_reolve_test);
}

TEST_CASE("Test external function resolve")
{
	frida_attach_manager man;
	void *addr = man.find_function_addr_by_name("malloc");
	REQUIRE(addr != nullptr);
	INFO("malloc addr resolved: " << (uintptr_t)addr);
	REQUIRE(addr == (void *)&malloc);
}
