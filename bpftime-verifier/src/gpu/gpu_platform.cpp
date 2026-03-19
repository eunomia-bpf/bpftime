#include "gpu_platform.hpp"

#include "crab_verifier.hpp"
#include "linux/linux_platform.hpp"
#include "platform-impl.hpp"
#include "spec_type_descriptors.hpp"

#include <linux/bpf.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace
{

using bpftime::GpuHelperArgumentSemantics;
using bpftime::GpuHelperBehavior;
using bpftime::GpuHelperEffectClass;
using bpftime::GpuHelperPrototype;
using bpftime::GpuHelperUniformity;

constexpr std::array<ebpf_argument_type_t, 5> ARGS_NONE = {
	EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE, EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<GpuHelperArgumentSemantics, 5> SEM_NONE = {
	GpuHelperArgumentSemantics::NONE, GpuHelperArgumentSemantics::NONE,
	GpuHelperArgumentSemantics::NONE, GpuHelperArgumentSemantics::NONE,
	GpuHelperArgumentSemantics::NONE,
};

constexpr std::array<ebpf_argument_type_t, 5> MAP_LOOKUP_ARGS = {
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP, EBPF_ARGUMENT_TYPE_PTR_TO_MAP_KEY,
	EBPF_ARGUMENT_TYPE_DONTCARE,   EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<ebpf_argument_type_t, 5> MAP_UPDATE_ARGS = {
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP,	     EBPF_ARGUMENT_TYPE_PTR_TO_MAP_KEY,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP_VALUE, EBPF_ARGUMENT_TYPE_ANYTHING,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<ebpf_argument_type_t, 5> MAP_DELETE_ARGS = {
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP, EBPF_ARGUMENT_TYPE_PTR_TO_MAP_KEY,
	EBPF_ARGUMENT_TYPE_DONTCARE,   EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<ebpf_argument_type_t, 5> GPU_PUTS_ARGS = {
	EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM,
	EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

// GPU helper out-parameters do not carry explicit size registers, so keep the
// PREVAIL-facing signature conservative here and recover the write semantics
// from semantic_argument_types during the SIMT passes.
constexpr std::array<ebpf_argument_type_t, 5> GPU_TRIPLE_OUT_ARGS = {
	EBPF_ARGUMENT_TYPE_ANYTHING, EBPF_ARGUMENT_TYPE_ANYTHING,
	EBPF_ARGUMENT_TYPE_ANYTHING, EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<ebpf_argument_type_t, 5> WARP_AGGREGATION_ARGS = {
	EBPF_ARGUMENT_TYPE_ANYTHING, EBPF_ARGUMENT_TYPE_ANYTHING,
	EBPF_ARGUMENT_TYPE_ANYTHING, EBPF_ARGUMENT_TYPE_DONTCARE,
	EBPF_ARGUMENT_TYPE_DONTCARE,
};

constexpr std::array<ebpf_argument_type_t, 5> GPU_PERF_EVENT_OUTPUT_ARGS = {
	EBPF_ARGUMENT_TYPE_ANYTHING,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP,
	EBPF_ARGUMENT_TYPE_ANYTHING,
	EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM,
	EBPF_ARGUMENT_TYPE_CONST_SIZE_OR_ZERO,
};

constexpr std::array<GpuHelperArgumentSemantics, 5> GPU_TRIPLE_U64_OUT = {
	GpuHelperArgumentSemantics::PTR_TO_U64_OUT,
	GpuHelperArgumentSemantics::PTR_TO_U64_OUT,
	GpuHelperArgumentSemantics::PTR_TO_U64_OUT,
	GpuHelperArgumentSemantics::NONE,
	GpuHelperArgumentSemantics::NONE,
};

constexpr std::array<GpuHelperArgumentSemantics, 5>
	CONSERVATIVE_UNKNOWN_GPU_OUT = {
		GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT,
		GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT,
		GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT,
		GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT,
		GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT,
	};

GpuHelperPrototype
make_helper(int32_t id, const char *name, ebpf_return_type_t return_type,
	    std::array<ebpf_argument_type_t, 5> prevail_argument_types,
	    GpuHelperUniformity return_uniformity,
	    std::array<GpuHelperArgumentSemantics, 5> semantic_argument_types =
		    SEM_NONE,
	    GpuHelperEffectClass effect_class = GpuHelperEffectClass::NONE,
	    GpuHelperBehavior behavior = GpuHelperBehavior::GENERIC)
{
	return GpuHelperPrototype{
		.id = id,
		.name = name,
		.return_type = return_type,
		.prevail_argument_types = prevail_argument_types,
		.return_uniformity = return_uniformity,
		.semantic_argument_types = semantic_argument_types,
		.effect_class = effect_class,
		.behavior = behavior,
	};
}

const std::map<int32_t, GpuHelperPrototype> &helper_table()
{
	static const std::map<int32_t, GpuHelperPrototype> table = {
		{ 1, make_helper(1, "bpf_map_lookup_elem",
				 EBPF_RETURN_TYPE_PTR_TO_MAP_VALUE_OR_NULL,
				 MAP_LOOKUP_ARGS, GpuHelperUniformity::UNIFORM,
				 SEM_NONE, GpuHelperEffectClass::NONE,
				 GpuHelperBehavior::MAP_LOOKUP) },
		{ 2, make_helper(2, "bpf_map_update_elem",
				 EBPF_RETURN_TYPE_INTEGER, MAP_UPDATE_ARGS,
				 GpuHelperUniformity::UNIFORM, SEM_NONE,
				 GpuHelperEffectClass::NONE,
				 GpuHelperBehavior::MAP_UPDATE) },
		{ 3, make_helper(3, "bpf_map_delete_elem",
				 EBPF_RETURN_TYPE_INTEGER, MAP_DELETE_ARGS,
				 GpuHelperUniformity::UNIFORM, SEM_NONE,
				 GpuHelperEffectClass::NONE,
				 GpuHelperBehavior::MAP_DELETE) },
		{ 14, make_helper(14, "bpf_get_current_pid_tgid",
				  EBPF_RETURN_TYPE_INTEGER, ARGS_NONE,
				  GpuHelperUniformity::UNIFORM) },
		{ 25,
		  make_helper(25, "perf_event_output", EBPF_RETURN_TYPE_INTEGER,
			      GPU_PERF_EVENT_OUTPUT_ARGS,
			      GpuHelperUniformity::UNIFORM) },
		{ 501,
		  make_helper(501, "ebpf_puts", EBPF_RETURN_TYPE_INTEGER,
			      GPU_PUTS_ARGS, GpuHelperUniformity::UNIFORM) },
		{ 502, make_helper(502, "bpf_get_globaltimer",
				   EBPF_RETURN_TYPE_INTEGER, ARGS_NONE,
				   GpuHelperUniformity::UNIFORM) },
		{ 503,
		  make_helper(503, "bpf_get_block_idx",
			      EBPF_RETURN_TYPE_INTEGER, GPU_TRIPLE_OUT_ARGS,
			      GpuHelperUniformity::UNIFORM,
			      GPU_TRIPLE_U64_OUT) },
		{ 504,
		  make_helper(504, "bpf_get_block_dim",
			      EBPF_RETURN_TYPE_INTEGER, GPU_TRIPLE_OUT_ARGS,
			      GpuHelperUniformity::UNIFORM,
			      GPU_TRIPLE_U64_OUT) },
		{ 505,
		  make_helper(505, "bpf_get_thread_idx",
			      EBPF_RETURN_TYPE_INTEGER, GPU_TRIPLE_OUT_ARGS,
			      GpuHelperUniformity::VARYING,
			      GPU_TRIPLE_U64_OUT) },
		{ 506,
		  make_helper(506, "bpf_gpu_membar", EBPF_RETURN_TYPE_INTEGER,
			      ARGS_NONE, GpuHelperUniformity::UNIFORM, SEM_NONE,
			      GpuHelperEffectClass::PROHIBITED_SYNC) },
		{ 507,
		  make_helper(507, "bpf_cuda_exit",
			      EBPF_RETURN_TYPE_INTEGER_OR_NO_RETURN_IF_SUCCEED,
			      ARGS_NONE, GpuHelperUniformity::UNIFORM) },
		{ 508,
		  make_helper(508, "bpf_get_grid_dim", EBPF_RETURN_TYPE_INTEGER,
			      GPU_TRIPLE_OUT_ARGS, GpuHelperUniformity::UNIFORM,
			      GPU_TRIPLE_U64_OUT) },
		{ 509,
		  make_helper(509, "bpf_get_sm_id", EBPF_RETURN_TYPE_INTEGER,
			      ARGS_NONE, GpuHelperUniformity::VARYING) },
		{ 510,
		  make_helper(510, "bpf_get_warp_id", EBPF_RETURN_TYPE_INTEGER,
			      ARGS_NONE, GpuHelperUniformity::UNIFORM) },
		{ 511,
		  make_helper(511, "bpf_get_lane_id", EBPF_RETURN_TYPE_INTEGER,
			      ARGS_NONE, GpuHelperUniformity::VARYING) },
		{ 520,
		  make_helper(520, "bpf_warp_ballot_sync_placeholder",
			      EBPF_RETURN_TYPE_INTEGER, WARP_AGGREGATION_ARGS,
			      GpuHelperUniformity::VARYING, SEM_NONE,
			      GpuHelperEffectClass::WARP_AGGREGATION) },
		{ 521,
		  make_helper(521, "bpf_warp_shfl_sync_placeholder",
			      EBPF_RETURN_TYPE_INTEGER, WARP_AGGREGATION_ARGS,
			      GpuHelperUniformity::VARYING, SEM_NONE,
			      GpuHelperEffectClass::WARP_AGGREGATION) },
	};
	return table;
}

EbpfHelperPrototype to_prevail_prototype(const GpuHelperPrototype &helper)
{
	EbpfHelperPrototype prototype{
		.name = helper.name,
		.return_type = helper.return_type,
		.argument_type = {
			helper.prevail_argument_types[0],
			helper.prevail_argument_types[1],
			helper.prevail_argument_types[2],
			helper.prevail_argument_types[3],
			helper.prevail_argument_types[4],
		},
		.reallocate_packet = false,
		.context_descriptor = nullptr,
	};
	return prototype;
}

struct bpf_load_map_def {
	uint32_t type;
	uint32_t key_size;
	uint32_t value_size;
	uint32_t max_entries;
	uint32_t map_flags;
	uint32_t inner_map_idx;
	uint32_t numa_node;
};

EbpfProgramType gpu_get_program_type(const std::string &section,
				     const std::string &path)
{
	if (section.starts_with("kprobe/") ||
	    section.starts_with("kretprobe/") ||
	    section.starts_with("uprobe") ||
	    section.starts_with("uretprobe/")) {
		return g_ebpf_platform_linux.get_program_type(section, path);
	}
	if (bpftime::is_gpu_section(section)) {
		return g_ebpf_platform_linux.get_program_type("kprobe/__gpu__",
							      path);
	}
	throw std::runtime_error("Unsupported GPU section: " + section);
}

EbpfHelperPrototype gpu_get_helper_prototype(int32_t helper_id)
{
	if (!bpftime::usable_helpers.contains(helper_id)) {
		throw std::runtime_error("Unusable helper: " +
					 std::to_string(helper_id));
	}
	if (auto it = bpftime::non_kernel_helpers.find(helper_id);
	    it != bpftime::non_kernel_helpers.end()) {
		return it->second;
	}
	if (const auto *helper =
		    bpftime::find_gpu_helper_prototype(helper_id)) {
		return to_prevail_prototype(*helper);
	}
	return get_helper_prototype_linux(helper_id);
}

bool gpu_is_helper_usable(int32_t helper_id)
{
	return bpftime::usable_helpers.contains(helper_id);
}

void gpu_parse_maps_section(std::vector<EbpfMapDescriptor> &descriptors,
			    const char *data, size_t map_def_size,
			    int map_count, const ebpf_platform_t *,
			    ebpf_verifier_options_t options)
{
	std::vector<bpf_load_map_def> normalized;
	std::vector<uint32_t> original_types;
	normalized.reserve(map_count);
	original_types.reserve(map_count);

	for (int i = 0; i < map_count; ++i) {
		bpf_load_map_def def{};
		memcpy(&def, data + i * map_def_size,
		       std::min(map_def_size, sizeof(def)));
		original_types.push_back(def.type);
		if (auto mapped = bpftime::try_get_gpu_map_type(def.type)) {
			def.type = mapped->platform_specific_type;
		}
		normalized.push_back(def);
	}

	const size_t before = descriptors.size();
	g_ebpf_platform_linux.parse_maps_section(
		descriptors, reinterpret_cast<const char *>(normalized.data()),
		sizeof(bpf_load_map_def), map_count, &g_ebpf_platform_linux,
		options);

	for (int i = 0; i < map_count; ++i) {
		descriptors[before + static_cast<size_t>(i)].type =
			original_types[static_cast<size_t>(i)];
	}
}

EbpfMapDescriptor &gpu_get_map_descriptor(int fd)
{
	for (auto &descriptor : global_program_info->map_descriptors) {
		if (descriptor.original_fd == fd) {
			return descriptor;
		}
	}
	if (auto it = bpftime::map_descriptors.find(fd);
	    it != bpftime::map_descriptors.end()) {
		return it->second;
	}
	throw std::runtime_error("Invalid map fd: " + std::to_string(fd));
}

EbpfMapType gpu_get_map_type(uint32_t platform_specific_type)
{
	if (auto mapped =
		    bpftime::try_get_gpu_map_type(platform_specific_type)) {
		return *mapped;
	}
	return g_ebpf_platform_linux.get_map_type(platform_specific_type);
}

void gpu_resolve_inner_map_references(std::vector<EbpfMapDescriptor> &maps)
{
	g_ebpf_platform_linux.resolve_inner_map_references(maps);
}

} // namespace

namespace bpftime
{

bool is_gpu_section(std::string_view section_name)
{
	return section_name.find("cuda__") != std::string_view::npos ||
	       section_name.find("rocm__") != std::string_view::npos ||
	       section_name.find("__memcapture") != std::string_view::npos;
}

bool is_gpu_map_type(uint32_t platform_specific_type)
{
	return platform_specific_type == 1501 ||
	       (platform_specific_type >= 1502 &&
		platform_specific_type <= 1513) ||
	       platform_specific_type == 1527;
}

std::optional<EbpfMapType> try_get_gpu_map_type(uint32_t platform_specific_type)
{
	if (platform_specific_type == 1501) {
		return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_HASH);
	}
	if (platform_specific_type >= 1502 && platform_specific_type <= 1513) {
		return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_ARRAY);
	}
	if (platform_specific_type == 1527) {
		return g_ebpf_platform_linux.get_map_type(BPF_MAP_TYPE_RINGBUF);
	}
	return std::nullopt;
}

const GpuHelperPrototype *find_gpu_helper_prototype(int32_t helper_id)
{
	const auto &helpers = helper_table();
	if (auto it = helpers.find(helper_id); it != helpers.end()) {
		return &it->second;
	}
	return nullptr;
}

GpuHelperPrototype get_gpu_helper_effects(int32_t helper_id)
{
	if (const auto *helper = find_gpu_helper_prototype(helper_id)) {
		return *helper;
	}

	EbpfHelperPrototype prototype{};
	if (auto it = non_kernel_helpers.find(helper_id);
	    it != non_kernel_helpers.end()) {
		prototype = it->second;
	} else {
		prototype = get_helper_prototype_linux(helper_id);
	}

	GpuHelperPrototype helper{
		.id = helper_id,
		.name = prototype.name,
		.return_type = prototype.return_type,
		.prevail_argument_types = {
			prototype.argument_type[0],
			prototype.argument_type[1],
			prototype.argument_type[2],
			prototype.argument_type[3],
			prototype.argument_type[4],
		},
		.return_uniformity = GpuHelperUniformity::UNIFORM,
		.semantic_argument_types = SEM_NONE,
		.effect_class = GpuHelperEffectClass::NONE,
		.behavior = GpuHelperBehavior::GENERIC,
	};
	if (helper_id >= 501) {
		helper.return_uniformity = GpuHelperUniformity::VARYING;
		helper.semantic_argument_types = CONSERVATIVE_UNKNOWN_GPU_OUT;
	}
	return helper;
}

ebpf_platform_t gpu_platform_spec{
	.get_program_type = &gpu_get_program_type,
	.get_helper_prototype = &gpu_get_helper_prototype,
	.is_helper_usable = &gpu_is_helper_usable,
	.map_record_size = sizeof(bpf_load_map_def),
	.parse_maps_section = &gpu_parse_maps_section,
	.get_map_descriptor = &gpu_get_map_descriptor,
	.get_map_type = &gpu_get_map_type,
	.resolve_inner_map_references = &gpu_resolve_inner_map_references,
};

} // namespace bpftime
