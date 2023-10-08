#include "bpftime-verifier.hpp"
#include "ebpf_base.h"
#include "helpers.hpp"
#include "linux/bpf.h"
#include "spec_type_descriptors.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <ostream>
#include <platform-impl.hpp>
#include <platform.hpp>
#include <set>
#include <stdexcept>
#include <linux/linux_platform.hpp>
#include <bpf/bpf.h>
#include <platform-impl.hpp>
using namespace bpftime;
static EbpfProgramType bpftime_get_program_type(const std::string &section,
						const std::string &path)
{
	if (section.starts_with("uprobe") || section.starts_with("uretprobe") ||
	    section.starts_with("tracepoint")) {
		return g_ebpf_platform_linux.get_program_type(section, path);
	} else {
		throw std::runtime_error(
			std::string("Unsupported section by bpftime: ") +
			section);
	}
}

static EbpfHelperPrototype bpftime_get_helper_prototype(int32_t helper_id)
{
	if (usable_helpers.contains(helper_id)) {
		if (auto itr = non_kernel_helpers.find(helper_id);
		    itr != non_kernel_helpers.end()) {
			return itr->second;
		} else {
			auto prototype = get_helper_prototype_linux(helper_id);
			return prototype;
		}
	} else {
		throw std::runtime_error(std::string("Unusable helper: ") +
					 std::to_string(helper_id));
	}
}
static bool bpftime_is_helper_usable(int32_t helper_id)
{
	return usable_helpers.contains(helper_id);
}
static void bpftime_parse_maps_section(std::vector<EbpfMapDescriptor> &,
				       const char *, size_t, int,
				       const struct ebpf_platform_t *,
				       ebpf_verifier_options_t)
{
	throw std::runtime_error(
		std::string("parse ELF with bpftime is not supported now"));
}

static EbpfMapDescriptor &bpftime_get_map_descriptor(int fd)
{
	if (auto itr = map_descriptors.find(fd); itr != map_descriptors.end()) {
		return itr->second;
	} else {
		throw std::runtime_error(std::string("Invalid map fd: ") +
					 std::to_string(fd));
	}
}
static EbpfMapType bpftime_get_map_type(uint32_t platform_specific_type)
{
	if (platform_specific_type == BPF_MAP_TYPE_HASH ||
	    platform_specific_type == BPF_MAP_TYPE_ARRAY ||
	    platform_specific_type == BPF_MAP_TYPE_RINGBUF) {
		return g_ebpf_platform_linux.get_map_type(
			platform_specific_type);
	} else {
		throw std::runtime_error(
			std::string("Unsupported map type by bpftime: ") +
			std::to_string(platform_specific_type));
	}
}
static void
bpftime_resolve_inner_map_reference(std::vector<EbpfMapDescriptor> &)
{
	throw std::runtime_error(
		std::string("inner maps is not supported by bpftime now"));
}
namespace bpftime
{

struct ebpf_platform_t bpftime_platform_spec {
	.get_program_type = &bpftime_get_program_type,
	.get_helper_prototype = &bpftime_get_helper_prototype,
	.is_helper_usable = &bpftime_is_helper_usable, .map_record_size = 0,
	.parse_maps_section = &bpftime_parse_maps_section,
	.get_map_descriptor = &bpftime_get_map_descriptor,
	.get_map_type = &bpftime_get_map_type,
	.resolve_inner_map_references = &bpftime_resolve_inner_map_reference
};

// Thread independent
thread_local std::set<int32_t> usable_helpers;
thread_local std::map<int, EbpfMapDescriptor> map_descriptors;
thread_local std::map<int32_t, EbpfHelperPrototype> non_kernel_helpers;
std::vector<EbpfMapDescriptor> get_all_map_descriptors()
{
	std::vector<EbpfMapDescriptor> result;

	for (const auto &[k, v] : map_descriptors) {
		result.push_back(v);
	}
	return result;
}

} // namespace bpftime
