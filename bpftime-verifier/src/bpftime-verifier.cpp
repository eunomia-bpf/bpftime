#include "asm_syntax.hpp"
#include "config.hpp"
#include <ebpf_base.h>
#include <linux/gpl/spec_type_descriptors.hpp>
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
	.no_simplify = false,
	.mock_map_fds = true,
	.strict = false,
	.print_line_info = false,
	.allow_division_by_zero = true,
	.setup_constraints = true,
};

namespace bpftime
{
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
					       size_t num_inst,
					       const std::string &section_name,std::vector<int> usable_helpers)
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
	prog.info.platform = &bpftime_platform_spec;
	prog.info.type.name = section_name;
	prog.info.type.context_descriptor = &g_tracepoint_descr;
	prog.info.type.platform_specific_data = 1;
	prog.info.type.is_privileged = false;
	prog.info.map_descriptors ; // What shoule be in here?
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
} // namespace bpftime
