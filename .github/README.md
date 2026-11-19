# bpftime CI

This directory holds the GitHub Actions plumbing for bpftime: reusable
workflows under `workflows/`, reusable composite actions under `actions/`,
and test driver scripts under `script/`.

## Runner selection

Jobs that need privileged Linux runners call the reusable
`workflows/pick-runner.yml`, which chooses between the organization's VM
runners and GitHub-hosted runners based on current capacity. Forks and
unavailable VM clusters automatically fall back to GitHub-hosted runners.
GPU jobs run directly on the self-hosted GPU runners and do not use the
picker.

## Build cache policy (ccache)

All compiled jobs enable ccache through the composite action
[`actions/ccache/action.yml`](actions/ccache/action.yml):

```yaml
      - name: Enable ccache
        uses: ./.github/actions/ccache
        with:
          variant: ${{ matrix.container }} # optional cache key discriminator
```

The action installs ccache if needed, restores and saves
`${{ github.workspace }}/.ccache` through `actions/cache`, and puts
ccache compiler launchers on `PATH`. Because the launchers masquerade as
`gcc`, `g++`, `clang`, versioned compilers (e.g. `gcc-13`) and the cross
toolchains, both CMake and plain Make builds use the cache without any
changes to build flags. `CCACHE_COMPILERCHECK=content` keeps results
correct across base-image updates.

Rules:

- Every new workflow that compiles C/C++ MUST call this action. Keep the
  cache directory inside the workspace; container jobs cannot rely on a
  writable `$HOME`.
- The `variant` input distinguishes images or toolchains within one
  workflow; do not invent a new value when an existing one (container
  image name, `gpu`, `arm64`, ...) already identifies the runner.
- Cache entries are capped at 500 MiB each and namespaced per workflow and
  variant. Do not raise the cap without checking repository cache quota.

## Never delete a build tree

Deleting a build directory (`rm -rf build`, `rm -rf build-*`) discards
object files, forces a full rebuild, and wastes the warm ccache. It is
forbidden in CI workflows.

When a job needs a second, differently configured build, use a separate
build directory instead:

```yaml
# first build in build/, MPK build in build-mpk/
cmake -Bbuild-mpk -DBPFTIME_ENABLE_MPK=YES ...
cmake --build build-mpk --target bpftime_runtime_tests -j"$(nproc)"
```

Existing applications of this rule:

- `test-runtime.yml`: default build in `build/`, the MPK build in
  `build-mpk/`.
- `test-attach.yml`: frida attach tests in `build/`, syscall-trace attach
  tests in `build-syscall/`.

Allowed exceptions: removing a nested submodule checkout before
re-initializing that submodule (for example
`rm -rf third_party/bpftool/libbpf` followed by
`git submodule update --init`) is an explicit submodule repair step, not a
build-tree deletion, and remains permitted. Removing log files and stale
shared-memory entries (`logs/*.log`, `$BPFTIME_SHARED_MEMORY_PATH/bpftime*`)
is unaffected as well.

## Adding or changing a workflow

- Keep path filters narrow so documentation-only changes do not start the
  full matrix.
- Use `concurrency` with `cancel-in-progress: true` to cancel superseded
  runs on the same ref.
- Set `fail-fast: false` on matrices so one image failure does not hide
  the results of the other images.
- Validate workflow changes locally with
  [actionlint](https://github.com/rhysd/actionlint) before pushing.
