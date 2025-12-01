#include "catch2/catch_test_macros.hpp"
#include "frida-gum.h"
#include "frida_register_def.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <cmath>
#include <string>
using namespace bpftime;
using namespace attach;

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_attach_with_back_trace__func5(uint64_t x)
{
	asm("");
	uint64_t result = 0;
	for (int i = 1; i <= x; i++) {
		result += (uint64_t)sqrt(i);
	}
	return result;
}

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_attach_with_back_trace__func4(uint64_t x)
{
	asm("");
	return __bpftime_test_attach_with_back_trace__func5(x) + 1;
}

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_attach_with_back_trace__func3(uint64_t x)
{
	asm("");
	return __bpftime_test_attach_with_back_trace__func4(x) + 2;
}

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_attach_with_back_trace__func2(uint64_t x)
{
	asm("");
	return __bpftime_test_attach_with_back_trace__func3(x) + 3;
}

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_attach_with_back_trace__func1(uint64_t x)
{
	asm("");
	return __bpftime_test_attach_with_back_trace__func2(x) + 4;
}

TEST_CASE("Test with backtrace")
{
	frida_attach_impl impl;
	bool invoked = false;
	impl.create_uprobe_at(
		(void *)&__bpftime_test_attach_with_back_trace__func5,
		[&](const pt_regs &regs) {
			invoked = true;
			auto stack = (std::vector<uint64_t> *)
					     impl.call_attach_specific_function(
						     "generate_stack", nullptr);
			for (int i = 0; i < 4; i++) {
				auto addr = stack->at(i);
				GumDebugSymbolDetails debug_details;
				REQUIRE(gum_symbol_details_from_address(
						(GumReturnAddress)addr,
						&debug_details) == true);
				SPDLOG_INFO("symbol name {}",
					    debug_details.symbol_name);
				auto expected_name = std::string(
					"__bpftime_test_attach_with_back_trace__func");
				REQUIRE(std::string(debug_details.symbol_name)
						.starts_with(expected_name));
			}

			delete stack;
		});
	REQUIRE(__bpftime_test_attach_with_back_trace__func1(100) == 635);
	REQUIRE(invoked == true);
}
