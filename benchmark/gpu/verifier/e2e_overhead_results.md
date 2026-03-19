# RQ5: End-to-End Verification Overhead

Date: 2026-03-18

## Question

How much time does GPU verification add to the policy load path, and how does that compare with CUDA PTX compilation, which happens anyway?

## Measurement Method

Workload and policy:

- CUDA workload: `example/gpu/cuda-counter/vec_add`
- Policy: `benchmark/gpu/verifier/.live_demo/live_safe_block_idx.bpf.o`
- This policy is intentionally simple: one CUDA probe, safe warp-uniform branch, known to pass `STRICT`.

Verifier modes measured on the client:

- `NO_VERIFY`
- default (`WARNING`, by leaving `BPFTIME_VERIFIER_LEVEL` unset)
- `STRICT`

Measurement setup:

- 5 runs per mode
- The server-side loader always used `BPFTIME_VERIFIER_LEVEL=NO_VERIFY`
  - reason: isolate GPU-side verifier overhead in the client attach path instead of mixing in generic userspace verifier cost
- `vec_add` runs forever, so each trial used `timeout 5s` and captured four correct output iterations before termination
- Expected client exit code was `124` because of the timeout wrapper

Metrics recorded from runtime logs:

- `GPU verifier elapsed`
  - direct self-time of the GPU verifier
- `PTX compile elapsed`
  - CUDA compilation cost for the rewritten PTX
- `PTX patch/load elapsed`
  - full patch + compile + module-load phase after verification
- `loader_total_ms`
  - server-side libbpf object load plus attach time

Important note: `PTX patch/load elapsed` already includes `PTX compile elapsed`.

## Results

Average of 5 runs:

| Mode | Avg verifier ms | Avg PTX compile ms | Avg PTX patch/load ms | Avg loader total ms | Correct iterations per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NO_VERIFY` | 0.000 | 18.409 | 461.876 | 0.260 | 4.0 |
| default (`WARNING`) | 2.824 | 20.316 | 462.135 | 0.252 | 4.0 |
| `STRICT` | 2.803 | 18.435 | 453.773 | 0.257 | 4.0 |

Median of 5 runs:

| Mode | Median verifier ms | Median PTX compile ms | Median PTX patch/load ms |
| --- | ---: | ---: | ---: |
| `NO_VERIFY` | 0.000 | 17.898 | 469.313 |
| default (`WARNING`) | 2.817 | 20.189 | 460.282 |
| `STRICT` | 2.803 | 18.387 | 453.364 |

Raw run data:

| Mode | Run | Verifier ms | PTX compile ms | PTX patch/load ms | Loader total ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NO_VERIFY` | 1 | 0.000 | 17.898 | 469.674 | 0.240 |
| `NO_VERIFY` | 2 | 0.000 | 18.472 | 469.313 | 0.240 |
| `NO_VERIFY` | 3 | 0.000 | 21.265 | 471.004 | 0.288 |
| `NO_VERIFY` | 4 | 0.000 | 16.811 | 452.365 | 0.286 |
| `NO_VERIFY` | 5 | 0.000 | 17.600 | 447.026 | 0.245 |
| default | 1 | 2.827 | 24.048 | 460.212 | 0.281 |
| default | 2 | 2.817 | 20.891 | 462.294 | 0.224 |
| default | 3 | 2.855 | 20.189 | 460.282 | 0.286 |
| default | 4 | 2.816 | 17.033 | 453.797 | 0.236 |
| default | 5 | 2.806 | 19.419 | 474.092 | 0.233 |
| `STRICT` | 1 | 2.812 | 18.444 | 455.244 | 0.227 |
| `STRICT` | 2 | 2.779 | 18.226 | 453.364 | 0.238 |
| `STRICT` | 3 | 2.759 | 18.387 | 455.829 | 0.276 |
| `STRICT` | 4 | 2.803 | 17.988 | 451.245 | 0.274 |
| `STRICT` | 5 | 2.861 | 19.129 | 453.184 | 0.270 |

## Interpretation

The best direct measurement of verification overhead is the instrumented `GPU verifier elapsed` timer:

- default mode: 2.824 ms average
- strict mode: 2.803 ms average

Comparison points:

- against PTX compile time:
  - default verifier time is about 13.9% of PTX compile time
  - strict verifier time is about 15.2% of PTX compile time
- against the full PTX patch/load phase:
  - verifier time is about 0.6% of that phase

So the practical conclusion is:

- verification adds a small but measurable cost, about 2.8 ms per policy load
- CUDA PTX compilation is still substantially more expensive, around 18 to 20 ms
- the overall rewritten-module patch/load phase dominates both, around 454 to 462 ms

## What Actually Changes When Verification Is Enabled

Server-side policy load cost does not materially change:

- loader total stays around 0.25 ms in all modes

The extra cost appears in the client CUDA attach path:

- `NO_VERIFY`: skip verifier, then patch/compile/load PTX
- default or `STRICT`: run verifier first, then patch/compile/load PTX

Because the overall patch/load phase has much larger absolute latency than the verifier, subtracting whole-run wall-clock times between modes is noisy. The explicit verifier timer is the reliable overhead number.

## Caveats

- This benchmark used the minimal single-probe safe policy, not the shipped `cuda_probe` example, because the packaged example still triggers `CUDA_ERROR_ILLEGAL_ADDRESS` after PTX injection on this branch.
- The server-side generic verifier was deliberately disabled for these measurements to isolate the GPU verifier. If both verifiers run in the real deployment path, total load overhead will be higher than the numbers reported here.
