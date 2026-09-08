#pragma once

#include "bpftime-verifier.hpp"
#include "uniformity_analysis.hpp"

#include <cstddef>
#include <map>
#include <string>

struct ebpf_inst;

namespace bpftime::verifier::gpu
{

// Verifier-informed admission of the automatic warp-leader execution
// transformation for one eBPF program.
//
// A program is eligible only when lowering its hook-site invocations to
// once-per-active-warp leader execution is observationally equivalent to the
// current per-thread execution. The analysis therefore requires the full GPU
// verification pipeline (PREVAIL, uniformity, SIMT safety) to accept the
// program and adds transformation-specific rules:
//  - every side-effect helper call must be idempotent under the warp: shared
//    map update/delete with warp-uniform key/value on a shared GPU map;
//  - per-lane append-style helpers (perf_event_output, trace_printk, puts),
//    per-thread map writes, and bpf_cuda_exit are excluded because a leader
//    execution cannot reproduce their per-lane observable semantics;
//  - atomic instructions are excluded because a leader cannot reproduce 32
//    distinct atomic increments on one uniform address;
//  - stores through pointers that may target per-thread map slots or unknown
//    memory are excluded for the same reason.
//
// Programs that are not eligible must keep the ordinary per-thread hook call
// path; the runtime decides via this result, not by skipping verification.
struct WarpExecutionEligibilityResult {
	bool eligible = false;
	// Human-readable ineligibility reason; empty when eligible.
	std::string reason;
};

WarpExecutionEligibilityResult evaluate_warp_execution_eligibility(
	const ebpf_inst *instructions, size_t num_instructions,
	const std::map<int, bpftime::verifier::BpftimeMapDescriptor> &maps);

} // namespace bpftime::verifier::gpu
