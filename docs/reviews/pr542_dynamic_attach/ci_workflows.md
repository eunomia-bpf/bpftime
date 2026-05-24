# CI / GitHub Actions Review Notes

Files:

- `.github/workflows/build-gcc13.yml`
- `.github/workflows/test-gpu-examples.yml`
- `.github/workflows/test-ptxpass.yml`
- `attach/nv_attach_impl/CMakeLists.txt` (build wiring relevant to CI)

## Blocking

### W1. Hard-coded internal proxy + global git config in workflow

- **Location**: `.github/workflows/test-ptxpass.yml:36-51`
- **Problem**:
  - Hard-codes `192.168.15.1:2345` proxy values.
  - Uses `git config --global` to set proxy.
- **Impact**:
  - Breaks portability (most runners are not on that network).
  - Leaks internal topology.
  - Pollutes self-hosted runners persistently (affects other jobs/repos).
- **Recommended fix**:
  - Use org/repo secrets or runner-level env; enable proxy conditionally (`if: env.HTTP_PROXY != ''`).
  - Avoid `--global`. Use per-command `git -c http.proxy=...` or a temporary config file.

## Should fix

### W2. `fetch-depth: 0` everywhere increases cost

- **Location**:
  - `.github/workflows/build-gcc13.yml:41-46`
  - `.github/workflows/test-gpu-examples.yml:45-49`
  - `.github/workflows/test-ptxpass.yml:52-56`
- **Problem**: Full history clone for PRs increases time/bandwidth with little benefit.
- **Recommendation**:
  - Prefer `fetch-depth: 1` (or minimal depth) unless tags/history are needed.
  - If tags are needed, use `fetch-tags: true` with shallow depth.

### W3. Submodule init is duplicated and inconsistent across jobs

- **Location**:
  - `.github/workflows/test-gpu-examples.yml` (build job initializes bpftool+ubpf; other job(s) may differ)
  - `.github/workflows/test-ptxpass.yml:57-65` does extra init after checkout
- **Problem**: Repeated `rm -rf` + recursive init can be expensive and inconsistent, leading to flakes.
- **Recommendation**:
  - Normalize a single approach (composite action or reusable workflow).
  - Only recurse-init the submodules that are actually required; consider shallow submodule clones if supported.

### W4. Hard-coded CUDA/LLVM paths in `test-ptxpass`

- **Location**: `.github/workflows/test-ptxpass.yml:80-87`
- **Problem**: `BPFTIME_CUDA_ROOT` and `LLVM_DIR` are pinned to specific host installs.
- **Recommendation**:
  - Detect CUDA root (similar to `test-gpu-examples.yml` logic) or accept env-provided paths.
  - Detect LLVM via `llvm-config --cmakedir` where possible.

### W5. Artifact executability assumptions are partial

- **Location**: `.github/workflows/test-gpu-examples.yml` (post-download `chmod +x` only covers some binaries)
- **Problem**: Only `bpftime`/`bpftimetool` are chmod-checked; other executables may regress later.
- **Recommendation**:
  - Define a single list of required executables and enforce `chmod +x` + `test -x` for each.

## Optional

### W6. Dynamic attach smoke tests depend on log substrings

- **Location**: `.github/workflows/test-gpu-examples.yml` (expected string checks)
- **Problem**: Log text changes can break CI without functional regression.
- **Recommendation**:
  - Prefer stable tokens/exit codes in CI mode, or parse structured output.

### W7. `attach/nv_attach_impl/CMakeLists.txt` dependency ordering clarity

- **Note**: If OpenSSL linkage is needed for test targets, consider keeping `find_package` and link directives close to the targets, or documenting why ordering is safe.

