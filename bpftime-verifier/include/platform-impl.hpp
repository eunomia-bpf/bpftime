#ifndef _PLATFORM_IMPL_HPP
#define _PLATFORM_IMPL_HPP

#include "spec_type_descriptors.hpp"
#include <cstdint>
#include <vector>
#include <platform.hpp>
namespace bpftime
{
extern ebpf_platform_t bpftime_platform_spec;
extern thread_local std::set<int32_t> usable_helpers;
extern thread_local std::map<int, EbpfMapDescriptor> map_descriptors;
std::vector<EbpfMapDescriptor> get_all_map_descriptors();
} // namespace bpftime

#endif
