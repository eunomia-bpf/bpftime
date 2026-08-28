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

#if defined(__linux__) && defined(__x86_64__)
asm(".pushsection .text\n"
    ".balign 8\n"
    ".global __bpftime_test_short_back_trace__marker\n"
    ".type __bpftime_test_short_back_trace__marker,@function\n"
    "__bpftime_test_short_back_trace__marker:\n"
    "ret\n"
    ".balign 8\n"
    ".size __bpftime_test_short_back_trace__marker,"
    ".-__bpftime_test_short_back_trace__marker\n"
    ".popsection\n");
extern "C" void __bpftime_test_short_back_trace__marker();

extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_test_short_back_trace__caller(uint64_t value)
{
	__bpftime_test_short_back_trace__marker();
	asm volatile("");
	return value + 1;
}
#endif

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
			auto stack = (std::vector<uint64_t> *)
					     impl.call_attach_specific_function(
						     "generate_stack", nullptr);
			REQUIRE_FALSE(stack->empty());
			if (!use_fuzzy_backtracer) {
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
			}

			delete stack;
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
				auto stack = (std::vector<uint64_t> *)
						     impl.call_attach_specific_function(
							     "generate_stack", nullptr);
				REQUIRE(stack != nullptr);
				for (const auto addr : *stack) {
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
				delete stack;
			});
		REQUIRE(__bpftime_test_stripped_back_trace__func1(1) == 16);
	}

	REQUIRE(fixture_frames[1] >= 3);
	REQUIRE(fixture_frames[1] > fixture_frames[0]);
}

#if defined(__linux__) && defined(__x86_64__)
TEST_CASE("Short uprobe exposes a context for stack capture")
{
	auto *marker = reinterpret_cast<const uint8_t *>(
		&__bpftime_test_short_back_trace__marker);
	REQUIRE(reinterpret_cast<uintptr_t>(marker) % alignof(uint64_t) == 0);
	REQUIRE(*marker == 0xc3);

	frida_attach_impl impl(false);
	bool invoked = false;
	bool found_caller = false;
	impl.create_uprobe_at(
		(void *)&__bpftime_test_short_back_trace__marker,
		[&](const pt_regs &) {
			invoked = true;
			auto *stack = static_cast<std::vector<uint64_t> *>(
				impl.call_attach_specific_function(
					"generate_stack", nullptr));
			REQUIRE(stack != nullptr);
			REQUIRE_FALSE(stack->empty());
			for (const auto address : *stack) {
				GumDebugSymbolDetails details;
				if (gum_symbol_details_from_address(
					    (GumReturnAddress)address, &details) &&
				    std::string(details.symbol_name).find(
					    "__bpftime_test_short_back_trace__caller") !=
					    std::string::npos)
					found_caller = true;
			}
			delete stack;
		});
	REQUIRE(__bpftime_test_short_back_trace__caller(41) == 42);
	REQUIRE(invoked);
	REQUIRE(found_caller);
}
#endif
