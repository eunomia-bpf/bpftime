#include "simt_safety_check.hpp"

#include "gpu_platform.hpp"

#include <sstream>

namespace
{

using bpftime::GpuHelperBehavior;
using bpftime::GpuHelperEffectClass;
using bpftime::verifier::gpu::PointerRegion;
using bpftime::verifier::gpu::SimtCheckOptions;
using bpftime::verifier::gpu::SimtSafetyError;
using bpftime::verifier::gpu::SimtSafetyResult;
using bpftime::verifier::gpu::Uniformity;
using bpftime::verifier::gpu::UniformityAnalysisResult;
using bpftime::verifier::gpu::UniformityState;
using bpftime::verifier::BpftimeMapDescriptor;

constexpr const char *BRANCH_CHECK = "Warp-Uniform Branch Conditions";
constexpr const char *PROHIBITED_HELPER_CHECK = "Prohibited Helpers";
constexpr const char *ATOMIC_CHECK =
	"Atomic Operations on Uniform Addresses";
constexpr const char *MAP_KEY_CHECK = "Map Update Key Uniformity";

bool is_lddw_continuation(const ebpf_inst *instructions, size_t pc)
{
	return pc > 0 && instructions[pc - 1].opcode == INST_OP_LDDW_IMM;
}

bool is_conditional_jump(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	if (cls != INST_CLS_JMP && cls != INST_CLS_JMP32) {
		return false;
	}
	return instruction.opcode != INST_OP_CALL &&
	       instruction.opcode != INST_OP_EXIT &&
	       instruction.opcode != INST_OP_JA;
}

bool is_call(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_CALL;
}

bool is_atomic(const ebpf_inst &instruction)
{
	return (instruction.opcode & INST_CLS_MASK) == INST_CLS_STX &&
	       (instruction.opcode & 0xe0) == 0xc0;
}

std::map<int, BpftimeMapDescriptor>
effective_map_descriptors(const SimtCheckOptions &options)
{
	if (!options.map_descriptors.empty()) {
		return options.map_descriptors;
	}
	return bpftime::verifier::get_map_descriptors();
}

void add_error(SimtSafetyResult &result, size_t pc, const char *check_name,
		 std::string message)
{
	result.passed = false;
	result.errors.push_back(SimtSafetyError{
		.instruction_index = pc,
		.check_name = check_name,
		.message = std::move(message),
	});
}

bool address_is_uniform(const UniformityState &state, uint8_t reg)
{
	if (state.regs[reg] != Uniformity::UNIFORM) {
		return false;
	}

	const auto &pointer = state.pointers[reg];
	if (pointer.region == PointerRegion::UNKNOWN) {
		return true;
	}
	if (pointer.offset_uniformity != Uniformity::UNIFORM) {
		return false;
	}
	if (pointer.region == PointerRegion::STACK &&
	    !pointer.constant_offset.has_value()) {
		return false;
	}
	return true;
}

Uniformity key_uniformity(const UniformityState &state,
			  const std::map<int, BpftimeMapDescriptor> &maps,
			  std::optional<int32_t> map_fd, uint8_t key_reg)
{
	size_t key_size = 0;
	if (map_fd.has_value()) {
		if (auto it = maps.find(*map_fd); it != maps.end()) {
			key_size = it->second.key_size;
		}
	}
	if (key_size == 0) {
		key_size =
			bpftime::verifier::gpu::infer_pointer_access_width(
				state, key_reg);
	}
	return bpftime::verifier::gpu::query_pointer_uniformity(state, key_reg,
								key_size);
}

} // namespace

namespace bpftime::verifier::gpu
{

std::string SimtSafetyResult::summary() const
{
	if (errors.empty()) {
		return "SIMT safety checks passed";
	}

	std::ostringstream stream;
	for (const auto &error : errors) {
		stream << error.check_name << " at instruction "
		       << error.instruction_index << ": " << error.message
		       << '\n';
	}
	return stream.str();
}

SimtSafetyResult check_simt_safety(const ebpf_inst *instructions,
				     size_t num_instructions,
				     const UniformityAnalysisResult &uniformity,
				     const SimtCheckOptions &options)
{
	SimtSafetyResult result;
	if (!uniformity.success) {
		add_error(result, 0, "Uniformity Analysis",
			  uniformity.error_message.empty() ?
				  std::string("uniformity analysis failed") :
				  uniformity.error_message);
		return result;
	}

	const auto maps = effective_map_descriptors(options);

	for (size_t pc = 0; pc < num_instructions; ++pc) {
		if (pc >= uniformity.states.size() ||
		    is_lddw_continuation(instructions, pc)) {
			continue;
		}
		if (!uniformity.reachable.empty() &&
		    (pc >= uniformity.reachable.size() ||
		     !uniformity.reachable[pc])) {
			continue;
		}

		const auto &instruction = instructions[pc];
		const auto &state = uniformity.states[pc];

		if (options.strict_uniformity && is_conditional_jump(instruction)) {
			const Uniformity left = state.regs[instruction.dst];
			const Uniformity right =
				(instruction.opcode & INST_SRC_REG) != 0 ?
					state.regs[instruction.src] :
					Uniformity::UNIFORM;
			if (left != Uniformity::UNIFORM ||
			    right != Uniformity::UNIFORM) {
				++result.varying_branch_count;
				add_error(result, pc, BRANCH_CHECK,
					  "branch predicate is lane-varying");
			}
		}

		if (is_atomic(instruction) &&
		    !address_is_uniform(state, instruction.dst)) {
			++result.varying_atomic_count;
			add_error(result, pc, ATOMIC_CHECK,
				  "atomic target address is not warp-uniform");
		}

		if (!is_call(instruction)) {
			continue;
		}

		const auto helper = bpftime::get_gpu_helper_effects(instruction.imm);
		if (!options.allow_prohibited_sync &&
		    helper.effect_class ==
			    bpftime::GpuHelperEffectClass::PROHIBITED_SYNC) {
			++result.prohibited_helper_count;
			add_error(result, pc, PROHIBITED_HELPER_CHECK,
				  std::string("helper ") + helper.name +
					  " is prohibited in SIMT mode");
		}

		if (helper.behavior == GpuHelperBehavior::MAP_UPDATE ||
		    helper.behavior == GpuHelperBehavior::MAP_DELETE) {
			const Uniformity key = key_uniformity(
				state, maps, state.map_fds[1], 2);
			if (key != Uniformity::UNIFORM) {
				++result.varying_map_key_count;
				add_error(result, pc, MAP_KEY_CHECK,
					  "map key bytes are lane-varying");
			}
		}

		if (helper.behavior == GpuHelperBehavior::MAP_UPDATE &&
		    state.regs[4] != Uniformity::UNIFORM) {
			add_error(result, pc, MAP_KEY_CHECK,
				  "map update flags are lane-varying");
		}
	}

	return result;
}

} // namespace bpftime::verifier::gpu
