#include "uniformity_analysis.hpp"

#include "bpftime-verifier.hpp"
#include "gpu_platform.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <sstream>
#include <string>

namespace
{

using bpftime::GpuHelperArgumentSemantics;
using bpftime::GpuHelperBehavior;
using bpftime::GpuHelperEffectClass;
using bpftime::GpuHelperPrototype;
using bpftime::GpuHelperUniformity;
using bpftime::verifier::BpftimeMapDescriptor;
using bpftime::verifier::gpu::PointerProvenance;
using bpftime::verifier::gpu::PointerRegion;
using bpftime::verifier::gpu::Uniformity;
using bpftime::verifier::gpu::UniformityAnalysisResult;
using bpftime::verifier::gpu::UniformityState;

constexpr int STACK_SIZE = 512;

bool is_lddw(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_LDDW_IMM;
}

bool is_lddw_continuation(const ebpf_inst *instructions, size_t pc)
{
	return pc > 0 && is_lddw(instructions[pc - 1]);
}

bool is_exit(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_EXIT;
}

bool is_call(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_CALL;
}

bool is_unconditional_jump(const ebpf_inst &instruction)
{
	return instruction.opcode == INST_OP_JA;
}

bool is_conditional_jump(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	if (cls != INST_CLS_JMP && cls != INST_CLS_JMP32) {
		return false;
	}
	return !is_call(instruction) && !is_exit(instruction) &&
	       !is_unconditional_jump(instruction);
}

bool is_mov_imm(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	return (cls == INST_CLS_ALU || cls == INST_CLS_ALU64) &&
	       (instruction.opcode & INST_ALU_OP_MASK) == INST_ALU_OP_MOV &&
	       (instruction.opcode & INST_SRC_REG) == 0;
}

bool is_mov_reg(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	return (cls == INST_CLS_ALU || cls == INST_CLS_ALU64) &&
	       (instruction.opcode & INST_ALU_OP_MASK) == INST_ALU_OP_MOV &&
	       (instruction.opcode & INST_SRC_REG) != 0;
}

bool is_alu_reg(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	if (cls != INST_CLS_ALU && cls != INST_CLS_ALU64) {
		return false;
	}
	return !is_mov_reg(instruction) &&
	       (instruction.opcode & INST_SRC_REG) != 0;
}

bool is_alu_imm(const ebpf_inst &instruction)
{
	const uint8_t cls = instruction.opcode & INST_CLS_MASK;
	if (cls != INST_CLS_ALU && cls != INST_CLS_ALU64) {
		return false;
	}
	return !is_mov_imm(instruction) &&
	       (instruction.opcode & INST_SRC_REG) == 0;
}

bool is_add_sub(const ebpf_inst &instruction)
{
	const uint8_t op = instruction.opcode & INST_ALU_OP_MASK;
	return op == INST_ALU_OP_ADD || op == INST_ALU_OP_SUB;
}

size_t access_width(uint8_t opcode)
{
	switch (opcode & INST_SIZE_MASK) {
	case INST_SIZE_B:
		return 1;
	case INST_SIZE_H:
		return 2;
	case INST_SIZE_W:
		return 4;
	case INST_SIZE_DW:
		return 8;
	default:
		return 0;
	}
}

bool in_stack_bounds(int32_t offset, size_t width)
{
	if (width == 0) {
		return false;
	}
	return offset >= -STACK_SIZE && offset <= -1 &&
	       offset + static_cast<int32_t>(width) - 1 <= -1;
}

bool is_unmodified_context_pointer(const PointerProvenance &pointer)
{
	return pointer.region == PointerRegion::CONTEXT &&
	       pointer.offset_uniformity == Uniformity::UNIFORM &&
	       pointer.constant_offset.has_value() &&
	       *pointer.constant_offset == 0;
}

size_t stack_index(int32_t relative_offset)
{
	return static_cast<size_t>(STACK_SIZE + relative_offset);
}

Uniformity helper_uniformity(GpuHelperUniformity helper_uniformity)
{
	return helper_uniformity == GpuHelperUniformity::VARYING ?
		       Uniformity::VARYING :
		       Uniformity::UNIFORM;
}

UniformityState make_entry_state()
{
	UniformityState state;
	state.regs[1] = Uniformity::UNIFORM;
	state.pointers[1] = PointerProvenance{
		.region = PointerRegion::CONTEXT,
		.constant_offset = 0,
		.offset_uniformity = Uniformity::UNIFORM,
		.pointee_uniformity = Uniformity::UNIFORM,
	};
	state.regs[10] = Uniformity::UNIFORM;
	state.pointers[10] = PointerProvenance{
		.region = PointerRegion::STACK,
		.constant_offset = 0,
		.offset_uniformity = Uniformity::UNIFORM,
	};
	return state;
}

void erase_overlapping_pointer_slots(UniformityState &state, int32_t offset,
					   size_t width)
{
	for (auto it = state.stack_pointer_slots.begin();
	     it != state.stack_pointer_slots.end();) {
		const int32_t slot_offset = it->first;
		const bool overlaps =
			slot_offset < offset + static_cast<int32_t>(width) &&
			offset < slot_offset + 8;
		if (overlaps) {
			it = state.stack_pointer_slots.erase(it);
		} else {
			++it;
		}
	}
}

void write_stack_uniformity(UniformityState &state, int32_t offset, size_t width,
			       Uniformity value)
{
	if (!in_stack_bounds(offset, width)) {
		return;
	}
	for (size_t i = 0; i < width; ++i) {
		state.stack_bytes[stack_index(offset + static_cast<int32_t>(i))] =
			value;
	}
}

Uniformity load_stack_uniformity(const UniformityState &state, int32_t offset,
				 size_t width)
{
	if (!in_stack_bounds(offset, width)) {
		return Uniformity::UNKNOWN;
	}

	Uniformity result = Uniformity::UNIFORM;
	for (size_t i = 0; i < width; ++i) {
		result = bpftime::verifier::gpu::join_uniformity(
			result,
			state.stack_bytes[stack_index(offset +
						     static_cast<int32_t>(i))]);
	}
	return result;
}

PointerProvenance join_pointer(const PointerProvenance &lhs,
				 const PointerProvenance &rhs)
{
	if (lhs.region == PointerRegion::UNKNOWN) {
		return rhs;
	}
	if (rhs.region == PointerRegion::UNKNOWN) {
		return lhs;
	}

	PointerProvenance result;
	result.region = lhs.region == rhs.region ? lhs.region :
						    PointerRegion::UNKNOWN;
	result.offset_uniformity = bpftime::verifier::gpu::join_uniformity(
		lhs.offset_uniformity, rhs.offset_uniformity);
	result.pointee_uniformity = bpftime::verifier::gpu::join_uniformity(
		lhs.pointee_uniformity, rhs.pointee_uniformity);

	if (lhs.constant_offset.has_value() && rhs.constant_offset.has_value() &&
	    lhs.constant_offset == rhs.constant_offset &&
	    result.offset_uniformity != Uniformity::VARYING) {
		result.constant_offset = lhs.constant_offset;
	} else if (lhs.constant_offset != rhs.constant_offset) {
		result.offset_uniformity = Uniformity::VARYING;
	}

	return result;
}

bool merge_state(UniformityState &target, const UniformityState &source)
{
	UniformityState merged = target;

	for (size_t i = 0; i < merged.regs.size(); ++i) {
		merged.regs[i] = bpftime::verifier::gpu::join_uniformity(
			merged.regs[i], source.regs[i]);
		merged.pointers[i] =
			join_pointer(merged.pointers[i], source.pointers[i]);
		if (merged.map_fds[i] != source.map_fds[i]) {
			merged.map_fds[i].reset();
		}
	}

	for (size_t i = 0; i < merged.stack_bytes.size(); ++i) {
		merged.stack_bytes[i] = bpftime::verifier::gpu::join_uniformity(
			merged.stack_bytes[i], source.stack_bytes[i]);
	}

	std::map<int32_t, PointerProvenance> merged_slots;
	for (const auto &[offset, pointer] : target.stack_pointer_slots) {
		auto it = source.stack_pointer_slots.find(offset);
		if (it == source.stack_pointer_slots.end()) {
			continue;
		}
		const auto joined = join_pointer(pointer, it->second);
		if (joined.region != PointerRegion::UNKNOWN) {
			merged_slots[offset] = joined;
		}
	}
	merged.stack_pointer_slots = std::move(merged_slots);

	if (merged == target) {
		return false;
	}
	target = std::move(merged);
	return true;
}

std::vector<size_t> successors(const ebpf_inst *instructions, size_t count,
				 size_t pc)
{
	auto jump_target = [&](int16_t off) -> std::optional<size_t> {
		const int64_t target =
			static_cast<int64_t>(pc) + 1 + static_cast<int64_t>(off);
		if (target < 0 || target >= static_cast<int64_t>(count)) {
			return std::nullopt;
		}
		return static_cast<size_t>(target);
	};

	if (pc >= count) {
		return {};
	}
	const auto &instruction = instructions[pc];
	if (is_lddw(instruction)) {
		return pc + 2 < count ? std::vector<size_t>{ pc + 2 } :
				       std::vector<size_t>{};
	}
	if (is_exit(instruction)) {
		return {};
	}
	if (is_unconditional_jump(instruction)) {
		if (auto target = jump_target(instruction.offset)) {
			return { *target };
		}
		return {};
	}
	if (is_conditional_jump(instruction)) {
		std::vector<size_t> result;
		if (pc + 1 < count) {
			result.push_back(pc + 1);
		}
		if (auto target = jump_target(instruction.offset)) {
			result.push_back(*target);
		}
		return result;
	}
	if (pc + 1 < count) {
		return { pc + 1 };
	}
	return {};
}

Uniformity query_key_uniformity(const UniformityState &state, uint8_t reg,
				 size_t width)
{
	return bpftime::verifier::gpu::query_pointer_uniformity(state, reg,
							      width);
}

size_t contiguous_stack_region_width(const UniformityState &state,
				     int32_t start_offset)
{
	size_t width = 0;
	for (int32_t offset = start_offset; in_stack_bounds(offset, 1);
	     ++offset) {
		if (state.stack_bytes[stack_index(offset)] ==
		    Uniformity::UNKNOWN) {
			break;
		}
		++width;
	}
	return width;
}

size_t key_access_width(const UniformityState &state,
			const std::map<int, BpftimeMapDescriptor> &maps,
			std::optional<int32_t> map_fd, uint8_t key_reg)
{
	if (map_fd.has_value()) {
		if (auto it = maps.find(*map_fd); it != maps.end() &&
		    it->second.key_size > 0) {
			return it->second.key_size;
		}
	}
	return bpftime::verifier::gpu::infer_pointer_access_width(state,
								 key_reg);
}

void clear_register(UniformityState &state, uint8_t reg)
{
	state.regs[reg] = Uniformity::UNKNOWN;
	state.pointers[reg] = {};
	state.map_fds[reg].reset();
}

Uniformity helper_return_uniformity(const GpuHelperPrototype &helper,
					 const UniformityState &state,
					 const std::map<int, BpftimeMapDescriptor> &maps)
{
	if (helper.effect_class == GpuHelperEffectClass::WARP_AGGREGATION) {
		return Uniformity::UNIFORM;
	}
	if (helper.behavior == GpuHelperBehavior::MAP_LOOKUP) {
		return query_key_uniformity(
			state, 2,
			key_access_width(state, maps, state.map_fds[1], 2));
	}
	return helper_uniformity(helper.return_uniformity);
}

UniformityState transfer(const ebpf_inst *instructions, size_t count, size_t pc,
			   const UniformityState &input,
			   const std::map<int, BpftimeMapDescriptor> &maps)
{
	UniformityState output = input;
	if (pc >= count || is_lddw_continuation(instructions, pc)) {
		return output;
	}

	const auto &instruction = instructions[pc];
	const uint8_t dst = instruction.dst;
	const uint8_t src = instruction.src;

	if (is_lddw(instruction)) {
		output.regs[dst] = Uniformity::UNIFORM;
		output.pointers[dst] = {};
		output.map_fds[dst].reset();
		if (instruction.src == 1) {
			output.map_fds[dst] = instruction.imm;
		}
		return output;
	}

	if (is_mov_imm(instruction)) {
		output.regs[dst] = Uniformity::UNIFORM;
		output.pointers[dst] = {};
		output.map_fds[dst].reset();
		return output;
	}

	if (is_mov_reg(instruction)) {
		output.regs[dst] = input.regs[src];
		output.pointers[dst] = input.pointers[src];
		output.map_fds[dst] = input.map_fds[src];
		return output;
	}

	if (is_alu_reg(instruction)) {
		output.regs[dst] = bpftime::verifier::gpu::join_uniformity(
			input.regs[dst], input.regs[src]);
		output.map_fds[dst].reset();
		if (is_add_sub(instruction) &&
		    input.pointers[dst].region != PointerRegion::UNKNOWN) {
			output.pointers[dst] = input.pointers[dst];
			output.pointers[dst].constant_offset.reset();
			output.pointers[dst].offset_uniformity =
				bpftime::verifier::gpu::join_uniformity(
					input.pointers[dst].offset_uniformity,
					input.regs[src]);
			output.regs[dst] = output.pointers[dst].offset_uniformity;
		} else {
			output.pointers[dst] = {};
		}
		return output;
	}

	if (is_alu_imm(instruction)) {
		output.regs[dst] = input.regs[dst];
		output.map_fds[dst].reset();
		if (is_add_sub(instruction) &&
		    input.pointers[dst].region != PointerRegion::UNKNOWN) {
			output.pointers[dst] = input.pointers[dst];
			if (output.pointers[dst].constant_offset.has_value()) {
				const int32_t delta =
					(instruction.opcode & INST_ALU_OP_MASK) ==
							INST_ALU_OP_SUB ?
						-instruction.imm :
						instruction.imm;
				output.pointers[dst].constant_offset =
					*output.pointers[dst].constant_offset + delta;
			}
		} else {
			output.pointers[dst] = {};
		}
		return output;
	}

	if ((instruction.opcode & INST_CLS_MASK) == INST_CLS_ST) {
		const size_t width = access_width(instruction.opcode);
		const auto &base_pointer = input.pointers[dst];
		if (base_pointer.region == PointerRegion::STACK &&
		    base_pointer.constant_offset.has_value() &&
		    base_pointer.offset_uniformity == Uniformity::UNIFORM) {
			const int32_t offset =
				*base_pointer.constant_offset + instruction.offset;
			erase_overlapping_pointer_slots(output, offset, width);
			write_stack_uniformity(output, offset, width,
					      Uniformity::UNIFORM);
		}
		return output;
	}

	if ((instruction.opcode & INST_CLS_MASK) == INST_CLS_STX) {
		const size_t width = access_width(instruction.opcode);
		const auto &base_pointer = input.pointers[dst];
		if (base_pointer.region == PointerRegion::STACK &&
		    base_pointer.constant_offset.has_value() &&
		    base_pointer.offset_uniformity == Uniformity::UNIFORM) {
			const int32_t offset =
				*base_pointer.constant_offset + instruction.offset;
			erase_overlapping_pointer_slots(output, offset, width);
			write_stack_uniformity(output, offset, width,
					      input.regs[src]);
			if (width == 8 && input.pointers[src].region !=
						  PointerRegion::UNKNOWN) {
				output.stack_pointer_slots[offset] =
					input.pointers[src];
			}
		}
		return output;
	}

	if ((instruction.opcode & INST_CLS_MASK) == INST_CLS_LDX) {
		const size_t width = access_width(instruction.opcode);
		const auto &base_pointer = input.pointers[src];
		output.map_fds[dst].reset();
		if (base_pointer.region == PointerRegion::STACK) {
			if (base_pointer.offset_uniformity == Uniformity::VARYING) {
				output.regs[dst] = Uniformity::VARYING;
				output.pointers[dst] = {};
				return output;
			}
			if (!base_pointer.constant_offset.has_value()) {
				output.regs[dst] = Uniformity::UNKNOWN;
				output.pointers[dst] = {};
				return output;
			}
			const int32_t offset =
				*base_pointer.constant_offset + instruction.offset;
			output.regs[dst] = load_stack_uniformity(input, offset,
							 width);
			output.pointers[dst] = {};
			if (width == 8) {
				if (auto it = input.stack_pointer_slots.find(offset);
				    it != input.stack_pointer_slots.end()) {
					output.pointers[dst] = it->second;
				}
			}
			return output;
		}

		if (is_unmodified_context_pointer(base_pointer)) {
			output.regs[dst] = Uniformity::UNIFORM;
			output.pointers[dst] = {};
			return output;
		}

		if (base_pointer.region == PointerRegion::MAP_VALUE) {
			if (base_pointer.offset_uniformity != Uniformity::UNIFORM ||
			    base_pointer.pointee_uniformity ==
				    Uniformity::UNKNOWN) {
				output.regs[dst] = Uniformity::VARYING;
			} else {
				output.regs[dst] =
					base_pointer.pointee_uniformity;
			}
			output.pointers[dst] = {};
			return output;
		}

		output.regs[dst] = Uniformity::VARYING;
		output.pointers[dst] = {};
		return output;
	}

	if (is_call(instruction)) {
		const GpuHelperPrototype helper =
			bpftime::get_gpu_helper_effects(instruction.imm);
		const Uniformity return_uniformity =
			helper_return_uniformity(helper, input, maps);

		for (uint8_t reg = 1; reg <= 5; ++reg) {
			clear_register(output, reg);
		}

		output.regs[0] = return_uniformity;
		output.map_fds[0].reset();
		output.pointers[0] = {};

		if (helper.behavior == GpuHelperBehavior::MAP_LOOKUP) {
			output.pointers[0] = PointerProvenance{
				.region = PointerRegion::MAP_VALUE,
				.constant_offset = 0,
				.offset_uniformity = Uniformity::UNIFORM,
				.pointee_uniformity = return_uniformity,
			};
		}

		for (size_t i = 0; i < helper.semantic_argument_types.size(); ++i) {
			const auto semantics = helper.semantic_argument_types[i];
			if (semantics != GpuHelperArgumentSemantics::PTR_TO_U64_OUT &&
			    semantics !=
				    GpuHelperArgumentSemantics::CONSERVATIVE_PTR_OUT) {
				continue;
			}
			const uint8_t arg_reg = static_cast<uint8_t>(i + 1);
			const auto &pointer = input.pointers[arg_reg];
			if (pointer.region != PointerRegion::STACK ||
			    !pointer.constant_offset.has_value() ||
			    pointer.offset_uniformity != Uniformity::UNIFORM) {
				continue;
			}
			erase_overlapping_pointer_slots(
				output, *pointer.constant_offset, 8);
			write_stack_uniformity(output, *pointer.constant_offset, 8,
					      semantics ==
							      GpuHelperArgumentSemantics::
								      CONSERVATIVE_PTR_OUT ?
						      Uniformity::VARYING :
						      helper_uniformity(
							      helper.return_uniformity));
		}

		return output;
	}

	return output;
}

} // namespace

namespace bpftime::verifier::gpu
{

UniformityState::UniformityState()
{
	regs.fill(Uniformity::UNKNOWN);
	pointers.fill(PointerProvenance{});
	map_fds.fill(std::nullopt);
	stack_bytes.fill(Uniformity::UNKNOWN);
}

Uniformity join_uniformity(Uniformity lhs, Uniformity rhs)
{
	if (lhs == Uniformity::UNKNOWN) {
		return rhs;
	}
	if (rhs == Uniformity::UNKNOWN) {
		return lhs;
	}
	if (lhs == rhs) {
		return lhs;
	}
	return Uniformity::VARYING;
}

const char *to_string(Uniformity uniformity)
{
	switch (uniformity) {
	case Uniformity::UNKNOWN:
		return "UNKNOWN";
	case Uniformity::UNIFORM:
		return "UNIFORM";
	case Uniformity::VARYING:
		return "VARYING";
	}
	return "UNKNOWN";
}

Uniformity query_stack_uniformity(const UniformityState &state,
				  int32_t start_offset, size_t width)
{
	return load_stack_uniformity(state, start_offset, width);
}

Uniformity query_pointer_uniformity(const UniformityState &state, uint8_t reg,
				    size_t width)
{
	if (reg >= state.pointers.size()) {
		return Uniformity::UNKNOWN;
	}

	const auto &pointer = state.pointers[reg];
	if (pointer.region == PointerRegion::STACK) {
		if (pointer.offset_uniformity == Uniformity::VARYING) {
			return Uniformity::VARYING;
		}
		if (!pointer.constant_offset.has_value()) {
			return Uniformity::UNKNOWN;
		}
		return query_stack_uniformity(state, *pointer.constant_offset, width);
	}

	if (pointer.region != PointerRegion::UNKNOWN &&
	    pointer.offset_uniformity == Uniformity::VARYING) {
		return Uniformity::VARYING;
	}

	return state.regs[reg];
}

size_t infer_pointer_access_width(const UniformityState &state, uint8_t reg)
{
	if (reg >= state.pointers.size()) {
		return 1;
	}

	const auto &pointer = state.pointers[reg];
	if (pointer.region == PointerRegion::STACK &&
	    pointer.offset_uniformity == Uniformity::UNIFORM &&
	    pointer.constant_offset.has_value()) {
		return std::max<size_t>(
			1, contiguous_stack_region_width(state,
							 *pointer.constant_offset));
	}
	return 1;
}

UniformityAnalysisResult analyze_uniformity(const ebpf_inst *instructions,
					      size_t num_instructions,
					      const std::map<int, BpftimeMapDescriptor> &maps)
{
	UniformityAnalysisResult result;
	result.states.resize(num_instructions);
	result.reachable.resize(num_instructions, false);

	if (instructions == nullptr) {
		result.error_message = "null instruction stream";
		return result;
	}
	if (num_instructions == 0) {
		result.success = true;
		return result;
	}

	std::vector<bool> has_state(num_instructions, false);
	std::vector<bool> queued(num_instructions, false);
	std::deque<size_t> worklist;

	result.states[0] = make_entry_state();
	has_state[0] = true;
	result.reachable[0] = true;
	queued[0] = true;
	worklist.push_back(0);

	while (!worklist.empty()) {
		const size_t pc = worklist.front();
		worklist.pop_front();
		queued[pc] = false;

		if (!has_state[pc] || is_lddw_continuation(instructions, pc)) {
			continue;
		}

		const UniformityState out =
			transfer(instructions, num_instructions, pc,
				 result.states[pc], maps);
		for (const auto succ : successors(instructions, num_instructions,
						  pc)) {
			if (succ >= num_instructions) {
				std::ostringstream error;
				error << "jump target out of bounds at instruction "
				      << pc;
				result.error_message = error.str();
				return result;
			}

			bool changed = false;
			if (!has_state[succ]) {
				result.states[succ] = out;
				has_state[succ] = true;
				result.reachable[succ] = true;
				changed = true;
			} else {
				changed = merge_state(result.states[succ], out);
			}
			if (changed && !queued[succ]) {
				queued[succ] = true;
				worklist.push_back(succ);
			}
		}
	}

	result.success = true;
	return result;
}

} // namespace bpftime::verifier::gpu
