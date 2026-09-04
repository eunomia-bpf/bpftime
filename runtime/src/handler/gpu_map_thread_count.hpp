/* SPDX-License-Identifier: MIT */
#ifndef BPFTIME_GPU_MAP_THREAD_COUNT_HPP
#define BPFTIME_GPU_MAP_THREAD_COUNT_HPP

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

namespace bpftime::detail
{
inline constexpr uint64_t MAX_GPU_MAP_THREAD_COUNT = 1048576;

[[nodiscard]] inline std::optional<uint64_t>
parse_gpu_map_thread_count(std::string_view value) noexcept
{
	if (value.empty()) {
		return std::nullopt;
	}

	uint64_t parsed = 0;
	const auto *begin = value.data();
	const auto *end = begin + value.size();
	const auto [next, error] = std::from_chars(begin, end, parsed, 10);
	if (error != std::errc{} || next != end || parsed == 0 ||
	    parsed > MAX_GPU_MAP_THREAD_COUNT) {
		return std::nullopt;
	}
	return parsed;
}

[[nodiscard]] inline std::optional<uint64_t>
resolve_gpu_map_thread_count(const char *environment_value,
			     uint64_t attribute_value) noexcept
{
	if (environment_value != nullptr) {
		return parse_gpu_map_thread_count(environment_value);
	}
	if (attribute_value == 0 ||
	    attribute_value > MAX_GPU_MAP_THREAD_COUNT) {
		return std::nullopt;
	}
	return attribute_value;
}
} // namespace bpftime::detail

#endif
