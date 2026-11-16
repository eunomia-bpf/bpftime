#include "catch2/catch_test_macros.hpp"
#include "frida_callback_boundary.hpp"
#include <stdexcept>

using bpftime::attach::frida_c_boundary;

TEST_CASE("frida C boundary returns the normal result when nothing throws",
	  "[frida][boundary]")
{
	int fn_calls = 0;
	int fb_calls = 0;
	int r = frida_c_boundary(
		[&]() -> int {
			++fn_calls;
			return 42;
		},
		[&]() -> int {
			++fb_calls;
			return -1;
		});
	CHECK(r == 42);
	CHECK(fn_calls == 1);
	CHECK(fb_calls == 0);
}

TEST_CASE("frida C boundary swallows std exceptions and runs the fallback",
	  "[frida][boundary]")
{
	int fb_calls = 0;
	int r = frida_c_boundary(
		[&]() -> int {
			throw std::runtime_error("boom");
		},
		[&]() -> int {
			++fb_calls;
			return -7;
		});
	CHECK(r == -7);
	CHECK(fb_calls == 1);
}

TEST_CASE("frida C boundary swallows non-std exceptions",
	  "[frida][boundary]")
{
	int r = frida_c_boundary([&]() -> int { throw 123; },
				 [&]() -> int { return -9; });
	CHECK(r == -9);
}

TEST_CASE("frida C boundary supports void callbacks", "[frida][boundary]")
{
	bool ran = false;
	bool fallback_ran = false;
	frida_c_boundary([&] { ran = true; },
			 [&] { fallback_ran = true; });
	CHECK(ran);
	CHECK_FALSE(fallback_ran);

	bool swallowed = false;
	frida_c_boundary([&]() -> void { throw std::logic_error("x"); },
			 [&]() -> void { swallowed = true; });
	CHECK(swallowed);
}

TEST_CASE("frida C boundary is noexcept so exceptions cannot cross the C frame",
	  "[frida][boundary]")
{
	auto fn = []() noexcept -> int { return 1; };
	auto fb = []() noexcept -> int { return 2; };
	static_assert(noexcept(frida_c_boundary(fn, fb)),
		      "the frida injection boundary must be noexcept");
	auto vfn = []() noexcept {};
	auto vfb = []() noexcept {};
	static_assert(noexcept(frida_c_boundary(vfn, vfb)),
		      "the void frida injection boundary must be noexcept");
	SUCCEED();
}
