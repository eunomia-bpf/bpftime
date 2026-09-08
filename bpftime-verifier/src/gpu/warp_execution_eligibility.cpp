#include "warp_execution_eligibility.hpp"

#include "gpu_platform.hpp"
#include "gpu_verifier.hpp"
#include <ebpf_vm_isa.hpp>
#include "simt_safety_check.hpp"

#include <optional>
#include <cstring>
#include <sstream>

namespace
{

using bpftime::GpuHelperBehavior;
using bpftime::verifier::BpftimeMapDescriptor;
using bpftime::verifier::gpu::PointerRegion;
using bpftime::verifier::gpu::SimtSafetyResult;
using bpftime::verifier::gpu::Uniformity;
using bpftime::verifier::gpu::UniformityAnalysisResult;
using bpftime::verifier::gpu::UniformityState;

constexpr uint32_t GPU_MAP_TYPE_GPU_HASH_MAP = 1501; // shared host-backed hash
constexpr uint32_t GPU_MAP_TYPE_PER_GPU_TD_ARRAY = 1502; // per-thread
constexpr uint32_t GPU_MAP_TYPE_GPU_ARRAY = 1503; // shared device array
constexpr uint32_t GPU_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY = 1504; // shared
constexpr uint32_t GPU_MAP_TYPE_PER_GPU_TD_ARRAY_HOST = 1512; // per-thread
constexpr uint32_t GPU_MAP_TYPE_GPU_ARRAY_HOST = 1513; // shared host array

bool is_lddw_continuation(const ebpf_inst *instructions, size_t pc)
{
	return pc > 0 && instructions[pc - 1].opcode == INST_OP_LDDW_IMM;
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

bool is_shared_gpu_map(uint32_t map_type)
{
	switch (map_type) {
	case GPU_MAP_TYPE_GPU_HASH_MAP:
	case GPU_MAP_TYPE_GPU_ARRAY:
	case GPU_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY:
	case GPU_MAP_TYPE_GPU_ARRAY_HOST:
		return true;
	default:
		return false;
	}
}

bool is_per_thread_gpu_map(uint32_t map_type)
{
	return map_type == GPU_MAP_TYPE_PER_GPU_TD_ARRAY ||
	       map_type == GPU_MAP_TYPE_PER_GPU_TD_ARRAY_HOST;
}

Uniformity pointer_uniformity(const UniformityState &state, uint8_t reg,
			      const std::map<int, BpftimeMapDescriptor> &maps,
			      std::optional<int32_t> map_fd,
			      size_t fallback_width)
{
	size_t width = fallback_width;
	if (map_fd.has_value()) {
		if (auto it = maps.find(*map_fd); it != maps.end()) {
			width = it->second.value_size;
		}
	}
	if (width == 0)
		width = 1;
	return bpftime::verifier::gpu::query_pointer_uniformity(state, reg,
							       width);
}

} // namespace

namespace bpftime::verifier::gpu
{

WarpExecutionEligibilityResult evaluate_warp_execution_eligibility(
	const ebpf_inst *instructions, size_t num_instructions,
	const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps)
{
	WarpExecutionEligibilityResult result;

	if (instructions == nullptr || num_instructions == 0) {
		result.reason = "empty instruction stream";
		return result;
	}

	const auto uniformity = analyze_uniformity(instructions,
						   num_instructions, maps);
	if (!uniformity.success) {
		result.reason = uniformity.error_message;
		return result;
	}

	const auto simt =
		check_simt_safety(instructions, num_instructions, uniformity,
				  maps);
	if (!simt.passed) {
		result.reason = simt.summary();
		return result;
	}

	auto reject = [&](size_t pc, const char *message) {
		std::ostringstream stream;
		stream << "warp execution ineligible at instruction " << pc
		       << ": " << message;
		result.reason = stream.str();
		return result;
	};

	for (size_t pc = 0; pc < num_instructions; ++pc) {
		if (is_lddw_continuation(instructions, pc))
			continue;
		if (pc >= uniformity.reachable.size() ||
		    !uniformity.reachable[pc])
			continue;
		const auto &instruction = instructions[pc];
		const auto &state = uniformity.states[pc];

		// A leader cannot reproduce 32 distinct atomic RMW side
		// effects on one uniform address.
		if (is_atomic(instruction)) {
			return reject(pc, "atomic side effects are lane-multiplicative");
		}

		const uint8_t instruction_class =
			instruction.opcode & INST_CLS_MASK;
		// Plain immediate and register stores alike: per-thread map
		// slots and unknown destinations have per-lane observable
		// effects a leader cannot reproduce.
		if ((instruction_class == INST_CLS_ST ||
		     instruction_class == INST_CLS_STX) && !is_atomic(instruction)) {
			// Plain stores: per-thread map slots and unknown
			// destinations have per-lane observable effects.
			const auto &base = state.pointers[instruction.dst];
			if (base.region == PointerRegion::UNKNOWN) {
				return reject(pc,
					      "store through an untracked pointer");
			}
			if (base.region == PointerRegion::MAP_VALUE &&
			    !base.may_target_shared_map) {
				return reject(
					pc,
					"store may target a per-thread map slot");
			}
		}

		if (!is_call(instruction))
			continue;

		const auto helper = bpftime::get_gpu_helper_effects(
			instruction.imm);
		switch (instruction.imm) {
		case 25: // bpf_perf_event_output appends one record per lane
			return reject(pc, "per-lane perf_event_output records cannot be deduplicated without changing the recorded event set");
		case 6: // trace_printk appends one host log line per lane
			return reject(pc, "per-lane trace_printk appends are not idempotent");
		case 501: // puts appends host output per lane
			return reject(pc, "per-lane puts appends are not idempotent");
		case 507: // cuda_exit would terminate only the leader's thread
			return reject(pc, "bpf_cuda_exit must run in every lane");
		default:
			break;
		}

		if (helper.behavior == GpuHelperBehavior::MAP_UPDATE ||
		    helper.behavior == GpuHelperBehavior::MAP_DELETE) {
			const auto map_fd = state.map_fds[1];
			if (!map_fd.has_value()) {
				return reject(pc,
					      "map update/delete target map cannot be resolved");
			}
			const auto map_it = maps.find(*map_fd);
			if (map_it == maps.end()) {
				return reject(pc,
					      "map update/delete target map is unknown to the verifier");
			}
			const auto map_type = map_it->second.type;
			if (is_per_thread_gpu_map(map_type)) {
				return reject(
					pc,
					"map update/delete targets a per-thread map slot");
			}
			if (!is_shared_gpu_map(map_type)) {
				return reject(pc,
					      "map update/delete target map is not a shared GPU map");
			}
			if (helper.behavior == GpuHelperBehavior::MAP_UPDATE) {
				const auto value_uniform =
					pointer_uniformity(state, 3, maps,
							  map_fd,
							  map_it->second.value_size);
				if (value_uniform != Uniformity::UNIFORM) {
					return reject(
						pc,
						"map update value bytes are not proven warp-uniform");
				}
			}
		}
	}

	result.eligible = true;
	return result;
}

} // namespace bpftime::verifier::gpu

namespace bpftime::verifier::gpu
{

std::optional<std::string> evaluate_warp_execution_eligibility(
	const void *raw_instructions, size_t num_instructions,
	const std::string &section_name,
	const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps)
{
	if (raw_instructions == nullptr || num_instructions == 0)
		return std::optional<std::string>(
			"warp execution ineligible: empty instruction stream");

	// Reuse the complete verification pipeline; the transformation is
	// only admitted on top of a fully verified program (no warning-mode
	// bypass).
	if (const auto verification_error = verify_gpu_program(
		    raw_instructions, num_instructions, section_name, maps);
	    verification_error.has_value()) {
		return std::optional<std::string>(
			"warp execution ineligible: GPU verification rejected the program: " +
			*verification_error);
	}

	std::vector<ebpf_inst> instructions(num_instructions);
	const auto *bytes = static_cast<const std::byte *>(raw_instructions);
	std::memcpy(instructions.data(), bytes,
		    num_instructions * sizeof(ebpf_inst));

	const auto result = evaluate_warp_execution_eligibility(
		instructions.data(), instructions.size(), maps);
	if (result.eligible)
		return std::nullopt;
	return std::optional<std::string>(result.reason);
}

} // namespace bpftime::verifier::gpu
