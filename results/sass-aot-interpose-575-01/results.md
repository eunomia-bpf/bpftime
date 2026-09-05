# Launch-boundary interposition results - driver-575 run 01 (2026-09-05)

Source: `results/sass-aot-interpose-575-01/raw.log` (RTX 5090, driver
575.57.08, device 0, runs: 5, iterations: 64). Each run has an uninstrumented
and an interposed phase; all 10 phases exited rc=0 and the log contains 320
"BPF-derived SASS callback executed" lines (5 interposed runs x 64
iterations, context value 42 each). Recorded as captured: no gates, no
hashes, no retries, no filtering, no result rejection.

## Medians over the 5 runs (ns, from the per-run `---` summaries)

| metric | uninstrumented | interposed | delta | ratio / overhead |
|---|---|---|---|---|
| cold total (iteration 0) | 16,205 | 25,007,647 | +24,991,442 | ~1543x |
| steady total per iteration (iters 1..63) | 5,137.5 | 32,101.5 | +26,964.0 | 6.25x / +524.8% |
| steady launch per iteration | 1,325.0 | 28,291.7 | +26,966.8 | ~21.4x |
| steady sync per iteration | 3,812.7 | 3,831.5 | +18.9 | ~1.005x |

Interposed cold includes the one-time BPF-to-SASS compilation at the first
launch. Steady medians are the median over 5 runs of the per-run sums over
iterations 1..63 divided by 63. The steady overhead is nearly all in the
launch path (the BPF-derived SASS callback inside the interposed
`cuLaunchKernel`); sync time is essentially unchanged.

## Per-run summaries from `raw.log` (ns)

| run | cold uninst. | cold interp. | steady total sum uninst. (63 iters) | steady total sum interp. (63 iters) |
|---|---|---|---|---|
| 1 | 16,581 | 24,295,094 | 323,104 | 2,065,173 |
| 2 | 15,835 | 25,252,440 | 326,456 | 2,022,397 |
| 3 | 16,571 | 24,947,108 | 322,460 | 1,975,215 |
| 4 | 16,205 | 25,007,647 | 327,695 | 2,038,445 |
| 5 | 15,522 | 25,684,415 | 323,664 | 2,007,665 |

## Scope

This is a **first-party launch-boundary BPF-to-SASS microbenchmark**: a
first-party `LD_PRELOAD` CUDA Driver API interposer executes a verified
BPF-derived SASS callback (strict GPU verifier -> ptxpass eBPF-to-PTX ->
ptxas, compiled once per process) from inside the interposed
`cuLaunchKernel` of the first-party fixture application (which launches its
own PTX-free, SASS-only cubin), in the application's own CUDA context,
before the original launch.

It is **not** llama.cpp Table 1, **not** third-party process interposition,
and **not** in-body SASS rewriting: the application's own binary, module,
and SASS are never modified.
