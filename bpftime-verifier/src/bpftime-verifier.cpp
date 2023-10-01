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
namespace bpftime
{
std::optional<std::string> verify_ebpf_program(const uint64_t *raw_inst,
					       size_t num_inst,
					       const std::string &section_name)
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
	prog.info.platform = &g_ebpf_platform_linux;
	prog.info.type.name = section_name;
	prog.info.type.context_descriptor = &g_tracepoint_descr;
	prog.info.type.platform_specific_data = 1;
	prog.info.type.is_privileged = false;
	std::vector<std::vector<std::string> > notes;
	auto unmarshal_result = unmarshal(prog, notes);
	if (std::holds_alternative<std::string>(unmarshal_result)) {
		return std::get<std::string>(unmarshal_result);
	}
	auto inst_seq = std::get<InstructionSeq>(unmarshal_result);
	ebpf_verifier_stats_t stats;
    stats.max_instruction_count = stats.total_unreachable = stats.total_warnings = 0;
	std::ostringstream message;
	auto result =
		ebpf_verify_program(message, inst_seq, prog.info,
				    &ebpf_verifier_default_options, &stats);
	if (result) {
		return {};
	} else {
		return message.str();
	}
}
} // namespace bpftime
