#ifndef _BPFTIME_VERIFIER_HPP
#define _BPFTIME_VERIFIER_HPP
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace bpftime
{
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
					       size_t num_inst,
					       const std::string &section_name,
					       std::vector<int> usable_helpers);
}

#endif
