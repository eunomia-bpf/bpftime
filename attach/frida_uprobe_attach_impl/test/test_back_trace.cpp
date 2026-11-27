#include "catch2/catch_test_macros.hpp"
#include "frida-gum.h"
#include "frida_register_def.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
using namespace bpftime;
using namespace attach;

extern "C" uint64_t __bpftime_test_stripped_back_trace__func1(uint64_t x);
extern "C" uint64_t __bpftime_test_stripped_back_trace__func5(uint64_t x);

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
	bool use_fuzzy_backtracer = false;
	SECTION("accurate")
	{
		use_fuzzy_backtracer = false;
	}
	SECTION("fuzzy")
	{
		use_fuzzy_backtracer = true;
	}
	frida_attach_impl impl(use_fuzzy_backtracer);
	bool invoked = false;
	impl.create_uprobe_at(
		(void *)&__bpftime_test_attach_with_back_trace__func5,
		[&](const pt_regs &regs) {
			invoked = true;
			stack_walk_data walk;
			REQUIRE(impl.call_attach_specific_function(
					"generate_stack", &walk) == &walk);
			REQUIRE(walk.count > 0);
			if (!use_fuzzy_backtracer) {
				for (size_t i = 0; i < 4; i++) {
					auto addr = walk.frames[i];
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
			}
		});
	REQUIRE(__bpftime_test_attach_with_back_trace__func1(100) == 635);
	REQUIRE(invoked == true);
}

TEST_CASE("Fuzzy backtracer recovers a call chain without unwind metadata")
{
	std::array<std::size_t, 2> fixture_frames{};
	for (const bool use_fuzzy_backtracer : { false, true }) {
		frida_attach_impl impl(use_fuzzy_backtracer);
		impl.create_uprobe_at(
			(void *)&__bpftime_test_stripped_back_trace__func5,
			[&](const pt_regs &) {
				stack_walk_data walk;
				REQUIRE(impl.call_attach_specific_function(
						"generate_stack", &walk) ==
					&walk);
				for (size_t i = 0; i < walk.count; ++i) {
					const auto addr = walk.frames[i];
					GumDebugSymbolDetails details;
					if (gum_symbol_details_from_address(
						    (GumReturnAddress)addr,
						    &details) &&
					    std::string(details.symbol_name)
						    .starts_with(
							    "__bpftime_test_stripped_back_trace__func")) {
						fixture_frames[use_fuzzy_backtracer]++;
					}
				}
			});
		REQUIRE(__bpftime_test_stripped_back_trace__func1(1) == 16);
	}

	REQUIRE(fixture_frames[1] >= 3);
	REQUIRE(fixture_frames[1] > fixture_frames[0]);
}
