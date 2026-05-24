# `nv_attach_impl` Review Notes

Files:

- `attach/nv_attach_impl/nv_attach_impl.cpp`
- `attach/nv_attach_impl/nv_attach_impl.hpp`
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`
- `attach/nv_attach_impl/nv_elf_introspect.cpp`
- `attach/nv_attach_impl/nv_elf_introspect.hpp`
- `attach/nv_attach_impl/trampoline_ptx.h`
- `attach/nv_attach_impl/trampoline/Makefile`

## Highlights (Good / Risk-reducing choices)

- **Fail-open to original launch** when bootstrap is incomplete or patched launch fails.
  - This avoids “wrong kernel executed” at the cost of observability gaps.
- **Use of `CUevent` to wait for in-flight patched kernels** during detach is directionally correct, reducing CUDA IPC UAF risk.

## Blocking

### N1. Late bootstrap can permanently “succeed” after failure

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp:1885-1896`
- **Problem**: `call_once` swallows exceptions and still sets `late_bootstrap_done=true`.
- **Impact**: Permanent silent “no data” after transient failures.
- **Fix**: See `blocking.md` (B2).

## Should fix

### N2. Late bootstrap state machine can get stuck in `started=true`

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp:1906-1933`
- **Problem**: `late_bootstrap_started` only resets when `late_bootstrap_done` is false; combined with N1, it may never re-run.
- **Impact**: Future attaches/refresh can’t recover.
- **Fix**: Make “done” dependent on success; or reset `started=false` on failure unconditionally.

### N3. Temporary directory leaks on some `cuobjdump` extraction paths

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp` (fatbin/PTX extraction paths; temporary workdir creation and early returns)
- **Problem**: Some early returns can bypass `remove_all(working_dir)`.
- **Impact**: `/tmp/bpftime-fatbin-work.*` accumulation.
- **Fix**: Use RAII/scope-exit cleanup, keep “debug keep dir” as an explicit opt-in branch.

### N4. Prefill `kernel_name -> CUfunction` may be ambiguous with duplicate names

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp` (prefill loop over loaded fatbins; `record_patched_kernel_function`)
- **Problem**: Same kernel name can exist in multiple modules/fatbins; mapping by name alone is non-unique.
- **Impact**: Potential “silent wrong instrumentation” (hooks fire but refer to the wrong module’s function).
- **Fix options**:
  - Key by `(module identity, kernel_name)` and only use name-only mapping as best-effort fallback.
  - Detect collisions and refuse/disable fallback mapping for that name (log clearly).

### N5. Host symbol cache is `once_flag` and not reset across sessions

- **Location**:
  - `attach/nv_attach_impl/nv_attach_impl.hpp` (host symbol cache once_flag)
  - Reset paths only reset late bootstrap once_flag
- **Problem**: In long-lived processes that load/unload modules between sessions, cached symbol ranges can go stale.
- **Impact**: `resolve_host_function_symbol` fallback may mis-resolve or fail.
- **Fix**: Use a resettable once-flag (`unique_ptr<once_flag>`) similar to late bootstrap, or rebuild cache on session transitions.

### N6. Symbol range matching is fragile when `st_size == 0`

- **Location**:
  - `attach/nv_attach_impl/nv_elf_introspect.cpp` (building symbol ranges)
  - `attach/nv_attach_impl/nv_attach_impl.cpp` (range match logic)
- **Problem**: `end == start` makes range checks ambiguous; may match too broadly depending on logic.
- **Impact**: Wrong symbol name fallback; debugging pain.
- **Fix**: For zero-size symbols, only accept `needle == start`, or bound by the next symbol start.

### N7. ELF section lookup relies on disk file matching the in-memory mapping

- **Location**: `attach/nv_attach_impl/nv_elf_introspect.cpp:205-237` (`find_section_in_memory`)
- **Problem**: Uses file section headers to infer memory addresses.
- **Impact**: In container/overlay/updated binaries, mismatches can occur.
- **Fix**: Prefer runtime `dl_iterate_phdr` PT_LOAD mapping from file offset -> vaddr, then locate section by offsets.

### N8. Resource lifecycle: `cuDevicePrimaryCtxRetain` needs symmetric release

- **Location**: `attach/nv_attach_impl/nv_attach_impl.cpp` (detach wait / context handling)
- **Problem**: Retain without release can raise refcount over repeated sessions.
- **Impact**: Long-lived process resource growth.
- **Fix**: Pair `cuDevicePrimaryCtxRelease` on all paths where retain succeeded.

## Optional

- **Trampoline generation fragility**:
  - `attach/nv_attach_impl/trampoline/Makefile` relies on grepping PTX text patterns; fragile against toolchain output changes.
- **`trampoline_ptx.h` fixed `sm_61`**:
  - Acceptable if the rest of the pipeline normalizes per-target SM; document that contract.

