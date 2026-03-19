#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

struct ebpf_inst;

namespace bpftime::verifier::gpu
{

enum class Uniformity {
	UNKNOWN = 0,
	UNIFORM,
	VARYING,
};

enum class PointerRegion {
	UNKNOWN = 0,
	STACK,
	CONTEXT,
	MAP_VALUE,
	OTHER,
};

struct PointerProvenance {
	PointerRegion region = PointerRegion::UNKNOWN;
	std::optional<int32_t> constant_offset {};
	Uniformity offset_uniformity = Uniformity::UNKNOWN;
	Uniformity pointee_uniformity = Uniformity::UNKNOWN;

	bool operator==(const PointerProvenance &) const = default;
};

struct UniformityState {
	std::array<Uniformity, 11> regs {};
	std::array<PointerProvenance, 11> pointers {};
	std::array<std::optional<int32_t>, 11> map_fds {};
	std::array<Uniformity, 512> stack_bytes {};
	std::map<int32_t, PointerProvenance> stack_pointer_slots {};

	UniformityState();

	bool operator==(const UniformityState &) const = default;
};

struct UniformityAnalysisResult {
	bool success = false;
	std::string error_message;
	std::vector<UniformityState> states;
	std::vector<bool> reachable;
};

Uniformity join_uniformity(Uniformity lhs, Uniformity rhs);

const char *to_string(Uniformity uniformity);

Uniformity query_stack_uniformity(const UniformityState &state,
				  int32_t start_offset, size_t width);

Uniformity query_pointer_uniformity(const UniformityState &state, uint8_t reg,
				    size_t width);

size_t infer_pointer_access_width(const UniformityState &state, uint8_t reg);

UniformityAnalysisResult analyze_uniformity(const ebpf_inst *instructions,
					      size_t num_instructions);

} // namespace bpftime::verifier::gpu
