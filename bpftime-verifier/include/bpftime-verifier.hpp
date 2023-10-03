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
namespace verifier
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

typedef enum {
	EBPF_RETURN_TYPE_INTEGER = 0,
	EBPF_RETURN_TYPE_PTR_TO_MAP_VALUE_OR_NULL,
	EBPF_RETURN_TYPE_INTEGER_OR_NO_RETURN_IF_SUCCEED,
	EBPF_RETURN_TYPE_UNSUPPORTED,
} bpftime_return_type_t;

typedef enum {
	EBPF_ARGUMENT_TYPE_DONTCARE = 0,
	EBPF_ARGUMENT_TYPE_ANYTHING, // All values are valid, e.g., 64-bit
				     // flags.
	EBPF_ARGUMENT_TYPE_CONST_SIZE,
	EBPF_ARGUMENT_TYPE_CONST_SIZE_OR_ZERO,
	EBPF_ARGUMENT_TYPE_PTR_TO_CTX,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP_OF_PROGRAMS,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP_KEY,
	EBPF_ARGUMENT_TYPE_PTR_TO_MAP_VALUE,
	EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM, // Memory must have been
						// initialized.
	EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM_OR_NULL,
	EBPF_ARGUMENT_TYPE_PTR_TO_WRITABLE_MEM,
	EBPF_ARGUMENT_TYPE_UNSUPPORTED,
} bpftime_argument_type_t;

struct BpftimeHelperProrotype {
	const char *name;

	// The return value is returned in register R0.
	bpftime_return_type_t return_type;

	// Arguments are passed in registers R1 to R5.
	bpftime_argument_type_t argument_type[5];
};

void set_available_helpers(const std::vector<int32_t> &helpers);

void set_non_kernel_helpers(
	const std::map<int32_t, BpftimeHelperProrotype> &protos);

void set_map_descriptors(const std::map<int, BpftimeMapDescriptor> &maps);
std::map<int, BpftimeMapDescriptor> get_map_descriptors();
} // namespace verifier
} // namespace bpftime

#endif
