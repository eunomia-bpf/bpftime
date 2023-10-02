#ifndef _BPFTIME_VERIFIER_HPP
#define _BPFTIME_VERIFIER_HPP
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace bpftime
{
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
					       size_t num_inst,
					       const std::string &section_name);

struct BpftimeMapDescriptor {
	int original_fd;
	uint32_t type; // Platform-specific type value in ELF file.
	unsigned int key_size;
	unsigned int value_size;
	unsigned int max_entries;
	unsigned int inner_map_fd;
};

void set_available_helpers(const std::vector<int32_t>& helpers);

void set_map_descriptors(const std::map<int, BpftimeMapDescriptor>& maps);

} // namespace bpftime

#endif
