// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>

namespace bpftime::attach
{
// Shared by the HIP helpers and the CPU listener. Only the control words are
// atomic; release/acquire ownership transfers protect the request and result.
inline constexpr unsigned rocm_max_maps = 64;
inline constexpr unsigned rocm_max_value_size = 4096;
enum : uint32_t { rocm_idle, rocm_request, rocm_response };
struct alignas(64) rocm_channel_data {
	uint32_t lock, state, stopped, map_count;
	uint32_t value_sizes[rocm_max_maps];
	uint32_t operation, map, key;
	uint64_t flags, result, timeout_ticks;
	alignas(8) uint8_t value[rocm_max_value_size];
};
} // namespace bpftime::attach
