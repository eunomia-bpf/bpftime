#include "asm_syntax.hpp"
#include "config.hpp"
#include "helpers.hpp"
#include <ebpf_base.h>
#include <iostream>
#include <linux/bpf_common.h>
#include <linux/gpl/spec_type_descriptors.hpp>
#include <ostream>
#include <platform.hpp>
#include <cstring>
#include <ebpf_vm_isa.hpp>
#include <spec_type_descriptors.hpp>
#include <bpftime-verifier.hpp>
#include <crab_verifier.hpp>
#include <asm_unmarshal.hpp>
#include <sstream>
#include <variant>
#include <platform-impl.hpp>
static ebpf_verifier_options_t verifier_options = {
	.check_termination = true,
	.assume_assertions = true,
	.print_invariants = true,
	.print_failures = true,
	.no_simplify = true,
	.mock_map_fds = false,
	.strict = false,
	.print_line_info = true,
	.allow_division_by_zero = false,
	.setup_constraints = false,
};

namespace bpftime
{
namespace verifier
{
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
					       size_t num_inst,
					       const std::string &section_name,
					       bool ignore_lddw_src_reg)
{
	raw_program prog;
	prog.filename = "BPFTIME_VERIFIER";
	prog.section = section_name;
	for (size_t i = 0; i < num_inst; i++) {
		ebpf_inst inst;
		static_assert(sizeof(inst) == sizeof(uint64_t), "");
		memcpy(&inst, &raw_inst[i], sizeof(inst));
		prog.prog.push_back(inst);
	}
	for (size_t i = 0; i < prog.prog.size(); i++) {
		if (i + 1 >= prog.prog.size())
			continue;
		auto &curr = prog.prog[i];
		auto &next = prog.prog[i + 1];
		// Workaround for ebpf-verifier not supporting lddw helpers
		// greater than 1
		// Replacing the two instructions with
		// dst1 = r10
		// dst1 = dst1 - 8
		if (ignore_lddw_src_reg) {
			if (curr.opcode == 0x18 && (curr.src == 2)) {
				// curr.src = 1;
				// next.imm = 0;
				curr.src = 10;
				curr.offset = 0;
				curr.imm = 0;
				curr.opcode = 0xbc;
				// next = curr;
				next.dst = curr.dst;
				next.src = 0;
				next.opcode = BPF_ALU | BPF_K | BPF_ADD;
				next.imm = -8;
				next.offset = 0;
			}
		}
	}
	prog.info = {
		.platform = &bpftime_platform_spec,
		.map_descriptors = get_all_map_descriptors(),
		.type = bpftime_platform_spec.get_program_type(section_name,
							       ""),
	};
	global_program_info = prog.info;
	std::vector<std::vector<std::string> > notes;
	auto unmarshal_result = unmarshal(prog, notes);
	if (std::holds_alternative<std::string>(unmarshal_result)) {
		return std::get<std::string>(unmarshal_result);
	}
	auto inst_seq = std::get<InstructionSeq>(unmarshal_result);
	ebpf_verifier_stats_t stats;
	stats.max_instruction_count = stats.total_unreachable =
		stats.total_warnings = 0;
	std::ostringstream message;
	auto result = ebpf_verify_program(message, inst_seq, prog.info,
					  &verifier_options, &stats);
	if (result) {
		return {};
	} else {
		return message.str();
	}
}

void set_available_helpers(const std::vector<int32_t> &helpers)
{
	usable_helpers.clear();
	for (auto x : helpers)
		usable_helpers.insert(x);
}
void set_map_descriptors(const std::map<int, BpftimeMapDescriptor> &maps)
{
	map_descriptors.clear();
	for (const auto &[k, v] : maps) {
		map_descriptors[k] = EbpfMapDescriptor{
			.original_fd = v.original_fd,
			.type = v.type,
			.key_size = v.key_size,
			.value_size = v.value_size,
			.max_entries = v.max_entries,
			.inner_map_fd = v.inner_map_fd,
		};
	}
}
void set_non_kernel_helpers(
	const std::map<int32_t, BpftimeHelperProrotype> &protos)
{
	non_kernel_helpers.clear();
	for (const auto &[k, v] : protos) {
		non_kernel_helpers[k] = EbpfHelperPrototype{
			.name = v.name,
			.return_type = (ebpf_return_type_t)(int)v.return_type,
			.argument_type = {
				 (ebpf_argument_type_t)(int)v.argument_type[0],
				 (ebpf_argument_type_t)(int)v.argument_type[1],
				 (ebpf_argument_type_t)(int)v.argument_type[2],
				 (ebpf_argument_type_t)(int)v.argument_type[3],
				 (ebpf_argument_type_t)(int)v.argument_type[4],
			},
			.reallocate_packet = false,
			.context_descriptor = nullptr
		};
	}
}
std::map<int, BpftimeMapDescriptor> get_map_descriptors()
{
	std::map<int, BpftimeMapDescriptor> result;
	for (const auto &[k, v] : map_descriptors) {
		result[k] = { .original_fd = v.original_fd,
			      .type = v.type,
			      .key_size = v.key_size,
			      .value_size = v.value_size,
			      .max_entries = v.max_entries,
			      .inner_map_fd = v.inner_map_fd };
	}
	return result;
}
} // namespace verifier
} // namespace bpftime
