#ifndef _PLATFORM_IMPL_HPP
#define _PLATFORM_IMPL_HPP

#include <cstdint>
#include <vector>
#include <platform.hpp>
namespace bpftime
{
extern ebpf_platform_t bpftime_platform_spec;
void set_available_helpers(const std::vector<int32_t> helpers);
} // namespace bpftime

#endif
