#pragma once

#include "helpers.hpp"
#include "platform.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace bpftime
{

enum class GpuHelperUniformity {
	UNIFORM,
	VARYING,
};

enum class GpuHelperArgumentSemantics {
	NONE,
	PTR_TO_U64_OUT,
	CONSERVATIVE_PTR_OUT,
};

enum class GpuHelperEffectClass {
	NONE,
	PROHIBITED_SYNC,
};

enum class GpuHelperBehavior {
	GENERIC,
	MAP_LOOKUP,
	MAP_UPDATE,
	MAP_DELETE,
};

struct GpuHelperPrototype {
	const char *name = "";
	ebpf_return_type_t return_type = EBPF_RETURN_TYPE_INTEGER;
	std::array<ebpf_argument_type_t, 5> prevail_argument_types{};
	GpuHelperUniformity return_uniformity = GpuHelperUniformity::UNIFORM;
	std::array<GpuHelperArgumentSemantics, 5> semantic_argument_types{};
	GpuHelperEffectClass effect_class = GpuHelperEffectClass::NONE;
	GpuHelperBehavior behavior = GpuHelperBehavior::GENERIC;
};

bool is_gpu_section(std::string_view section_name);

std::optional<EbpfMapType>
try_get_gpu_map_type(uint32_t platform_specific_type);

const GpuHelperPrototype *find_gpu_helper_prototype(int32_t helper_id);

GpuHelperPrototype get_gpu_helper_effects(int32_t helper_id);

extern ebpf_platform_t gpu_platform_spec;

} // namespace bpftime
