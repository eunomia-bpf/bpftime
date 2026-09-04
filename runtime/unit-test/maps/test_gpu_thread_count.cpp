#include "handler/gpu_map_thread_count.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <optional>
#include <string_view>

using bpftime::detail::MAX_GPU_MAP_THREAD_COUNT;
using bpftime::detail::parse_gpu_map_thread_count;
using bpftime::detail::resolve_gpu_map_thread_count;

TEST_CASE("GPU map thread-count parser accepts only bounded decimal input",
	  "[gpu][map][thread-count]")
{
	const auto require_value = [](std::string_view input,
				      uint64_t expected) {
		const auto parsed = parse_gpu_map_thread_count(input);
		REQUIRE(parsed.has_value());
		REQUIRE(*parsed == expected);
	};

	require_value("1", 1);
	require_value("00042", 42);
	require_value("1048576", MAX_GPU_MAP_THREAD_COUNT);

	for (const std::string_view invalid : {
		     "", "0", "1048577", "-1", "+1", " 1", "1 ",
		     "1x", "0x10", "18446744073709551615",
		     "18446744073709551616" }) {
		CAPTURE(invalid);
		REQUIRE_FALSE(parse_gpu_map_thread_count(invalid).has_value());
	}
}

TEST_CASE("GPU map thread-count resolution fails closed",
	  "[gpu][map][thread-count]")
{
	REQUIRE(resolve_gpu_map_thread_count(nullptr, 1024) ==
		std::optional<uint64_t>{ 1024 });
	REQUIRE_FALSE(resolve_gpu_map_thread_count(nullptr, 0).has_value());
	REQUIRE_FALSE(
		resolve_gpu_map_thread_count(nullptr,
					     MAX_GPU_MAP_THREAD_COUNT + 1)
			.has_value());
	REQUIRE(resolve_gpu_map_thread_count("22528", 1024) ==
		std::optional<uint64_t>{ 22528 });
	REQUIRE_FALSE(
		resolve_gpu_map_thread_count("-1", 1024).has_value());
	REQUIRE_FALSE(
		resolve_gpu_map_thread_count("12garbage", 1024).has_value());
}
