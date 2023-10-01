#include "helpers.hpp"
#include "spec_type_descriptors.hpp"
#include <cassert>
#include <platform-impl.hpp>
#include <platform.hpp>
#include <set>
#include <stdexcept>
#include <linux/linux_platform.hpp>
// Thread dependent
thread_local std::set<int32_t> usable_helpers;

static EbpfProgramType bpftime_get_program_type(const std::string &section,
						const std::string &path)
{
}

static EbpfHelperPrototype bpftime_get_helper_prototype(int32_t helper_id)
{
	if (usable_helpers.contains(helper_id)) {
		return get_helper_prototype_linux(helper_id);
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

static EbpfMapDescriptor &bpftime_get_map_descriptor(int)
{
}
static EbpfMapType bpftime_get_map_type(uint32_t)
{
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

void set_available_helpers(const std::vector<int32_t> helpers)
{
	usable_helpers.clear();
	for (auto x : helpers)
		usable_helpers.insert(x);
}
} // namespace bpftime
