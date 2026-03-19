| Pattern | Representative Input | No Verification | Standard PREVAIL (design) | SIMT-aware (measured) |
| --- | --- | --- | --- | --- |
| Varying branch condition | gpu_unsafe_programs/varying_branch.bpf.c | MISS | MISS | CATCH |
| Prohibited helper (membar) | gpu_unsafe_programs/prohibited_helper.bpf.c | MISS | MISS | CATCH |
| Varying atomic address | gpu_unsafe_programs/varying_atomic.bpf.c | MISS | MISS | CATCH |
| Varying map key | gpu_unsafe_programs/varying_map_key.bpf.c | MISS | MISS | CATCH |
| Helper-call budget exceeded | gpu_unsafe_programs/resource_exceeded.bpf.c | MISS | MISS | CATCH |
| Memory safety (null deref) | builtin:null_deref | MISS | CATCH | MISS |
| Division by zero | builtin:division_by_zero | MISS | CATCH | MISS |
| Unbounded loop (self-loop) | builtin:resource_exceeded | MISS | CATCH | MISS |

Notes:
- `No Verification` is the baseline with the verifier disabled, so unsafe programs are not intercepted.
- `Standard PREVAIL (design)` is design-based coverage, not a measurement on GPU object files.
- `Unbounded loop (self-loop)` uses builtin `resource_exceeded`; it is distinct from `gpu_unsafe_programs/resource_exceeded.bpf.c`, which models helper-call budget exhaustion.
