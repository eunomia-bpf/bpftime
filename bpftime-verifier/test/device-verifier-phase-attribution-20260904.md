# GPU verifier phase-attribution preflight

## Outcome

This CPU-only, two-arm preflight localizes the previously observed 4,096-
instruction scaling reversal to PREVAIL. It does not show that branches make
verification faster in general, and it is not a repeated paper result.

Both fresh processes accepted the frozen programs. The processes ran serially,
were pinned to CPU 23, and had `CUDA_VISIBLE_DEVICES` empty and `LD_PRELOAD`
unset. The probe and verifier were rebuilt after commit `c1c4cf6` in Release
mode. The internal phase timer was enabled only for these diagnostic calls.

| 4,096-instruction arm | Internal wall total, ms | PREVAIL, ms (% total) | Uniformity, ms | SIMT, ms | Minor faults |
|---|---:|---:|---:|---:|---:|
| Linear (4,092 adds) | 1,909.423 | 1,907.221 (99.885%) | 2.170 | 0.009 | 932,625 |
| Uniform diamonds (2,046 branches, 2,046 adds) | 586.493 | 583.949 (99.566%) | 2.491 | 0.029 | 44,503 |

The diamonds/linear ratio is 0.3072 for the internal total and 0.3062 for
PREVAIL alone. PREVAIL differs by 1,323.271 ms, while the internal totals differ
by 1,322.930 ms. Thus PREVAIL accounts for 100.026% of the gap: uniformity,
SIMT, copying, and validation collectively make the diamond arm about 0.34 ms
slower rather than explaining its lower total. Process-CPU attribution agrees:
PREVAIL is 99.885% of the linear total and 99.567% of the diamond total.

The result rules out gpubpf's new uniformity analysis and SIMT safety scan as
the cause of this crossover. Source inspection shows that the PREVAIL interval
includes shadow-program construction, unmarshalling, CFG preparation, forward
abstract interpretation, and report construction. The linear constructor also
executes twice as many arithmetic abstract-domain operations as the diamond
constructor. Its 20.96x higher minor-fault count points to substantially
different PREVAIL allocation/state behavior. This is a plausible explanation,
not a sub-PREVAIL causal attribution; isolating one PREVAIL internal routine
would require another instrumented experiment.

## Validity and scope

- Internal wall/process-CPU totals differ by less than 0.04% in each serialized
  call; neither call had a major fault or voluntary context switch.
- All five phase bits were present and all phase values were nonnegative.
- The sum of measured phases leaves only 0.012 ms of unclassified internal wall
  time in each arm.
- The probe's outer wall clock uses `CLOCK_MONOTONIC_RAW`, whereas the new
  internal wall timer uses `steady_clock`; use the internal total for phase
  shares rather than subtracting the two clock domains.
- An earlier exploratory invocation mistakenly ran both CPU-23-pinned arms at
  the same time and showed scheduler contention. It was rejected before this
  serial preflight and is not used above.
- This is a dependency-level diagnosis of the completed scaling experiment,
  not evidence about GPU execution, verifier soundness, attach/JIT latency, or
  production-policy distributions.

## Commands

```sh
cmake -S /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-scaling \
  -B /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-table1-575
cmake --build /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build \
  --target verifier_scaling_probe bpftime_verifier_tests --parallel 2
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  BPFTIME_GPU_VERIFIER_PHASE_TIMING=1 taskset -c 23 \
  /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
  --family linear --instructions 4096 --require-cpu 23
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  BPFTIME_GPU_VERIFIER_PHASE_TIMING=1 taskset -c 23 \
  /home/yunwei37/workspace/gpu/bpftime-device-verifier-scaling-build/verifier_scaling_probe \
  --family diamonds --instructions 4096 --require-cpu 23
```

## Raw records

```json
{"family":"linear","outer":{"elapsed_ns":1909415571,"process_cpu_ns":1908698438,"minor_faults":932625,"major_faults":0,"voluntary_context_switches":0,"involuntary_context_switches":16},"phase":{"schema":"bpftime-gpu-verifier-phase-timing-v1","instruction_count":4096,"map_count":0,"accepted":true,"phase_mask":31,"wall_ns":{"input_copy":10796,"validation":180,"prevail":1907220792,"uniformity":2169519,"simt":9459,"total":1909423053},"process_cpu_ns":{"input_copy":10761,"validation":178,"prevail":1906458723,"uniformity":2169618,"simt":9478,"total":1908659902}}}
{"family":"diamonds","outer":{"elapsed_ns":586515598,"process_cpu_ns":586512511,"minor_faults":44503,"major_faults":0,"voluntary_context_switches":0,"involuntary_context_switches":1},"phase":{"schema":"bpftime-gpu-verifier-phase-timing-v1","instruction_count":4096,"map_count":0,"accepted":true,"phase_mask":31,"wall_ns":{"input_copy":10799,"validation":271,"prevail":583949433,"uniformity":2491439,"simt":29075,"total":586492898},"process_cpu_ns":{"input_copy":10835,"validation":268,"prevail":583930627,"uniformity":2491689,"simt":29100,"total":586472870}}}
```

## Checks and independent review

The complete CPU verifier test binary passed 139 assertions in 24 test cases.
The subprocess timing test separately exercised unset, `0`, and exact-`1`
settings; it checked accepted, PREVAIL-rejected, SIMT-rejected, null, empty,
oversized, and unsupported-map paths and their completion masks.

A separate read-only result check independently recomputed the phase shares,
ratios, gaps, and residuals from the two raw records and found no arithmetic or
validity blocker for PREVAIL-level attribution. It also confirmed the scope as
a one-sample-per-arm dependency preflight: the records cannot identify a
PREVAIL subroutine or support a repeated paper-facing performance claim.

A read-only implementation review found the default-off design feasible
without changing the public API or verifier result. It required immutable
process-start opt-in, per-call state, RAII closure on early returns, preserved
`errno`, one versioned standard-error record, and serialized attribution; the
implementation and tests cover those requirements. A corrected read-only
OpenCode invocation using the local `spark-gateway/qwen3.8-27b-nvfp4-200k`
model produced no events and timed out after 240 seconds, so it supplied no
verdict and is not represented as a pass.
