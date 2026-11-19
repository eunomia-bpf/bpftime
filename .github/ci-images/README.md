# CI base images

The test workflows run inside container images that previously came from
the externally maintained `manjusakalza/bpftime-base-image` (last updated
February 2024) and a vanilla openEuler image, so every job re-installed
its toolchain and dependencies at runtime. The Dockerfiles here bake
those packages into images published in the project's own GitHub
Container Registry namespace.

## Images

| Dockerfile | Image | Replaces |
| --- | --- | --- |
| `Dockerfile.ubuntu-2204` | `ghcr.io/eunomia-bpf/ci-ubuntu-2204` | `manjusakalza/bpftime-base-image:ubuntu-2204` |
| `Dockerfile.fedora-39` | `ghcr.io/eunomia-bpf/ci-fedora-39` | `manjusakalza/bpftime-base-image:fedora-39` |
| `Dockerfile.openeuler-2403` | `ghcr.io/eunomia-bpf/ci-openeuler-2403` | `openeuler/openeuler:24.03-lts-sp1` plus per-job installs |

Each image contains the union of packages the corresponding workflows
install today (compilers, CMake, LLVM/clang, Boost, OpenSSL, elfutils,
lcov, bpftrace where available, debugging tools) and `ccache` for the
[ccache composite action](../actions/ccache/action.yml). The conda
setup for `libabseil`/`libprotobuf` stays in the workflows for now; it
is not baked in.

## Publishing

There is no automatic trigger. From the GitHub UI run
**Actions → Build CI base images → Run workflow**:

- `images`: `all` or one specific image;
- `push`: keep enabled to publish, or disable to validate a Dockerfile
  change with a build-only run.

Each pushed image gets two tags:

- `ghcr.io/eunomia-bpf/<image>:<commit-sha>` — immutable, pin workflows
  to this once validated;
- `ghcr.io/eunomia-bpf/<image>:latest` — moving tag for exploratory
  runs.

Publishing uses the workflow's built-in `GITHUB_TOKEN` with
`packages: write`; the first run creates the packages under the
`eunomia-bpf` organization. The packages inherit the repository's
visibility; make them public in the package settings if anonymous pulls
are needed.

## Cutover

Switching the workflows is deliberately a separate, reviewable change:

1. Dispatch a build for the changed image with `push=false` to validate
   the Dockerfile, then a pushed build.
2. Update the `container.image` references in the affected workflows
   from `manjusakalza/bpftime-base-image:*` / vanilla openEuler to the
   matching `ghcr.io/eunomia-bpf/ci-*:<commit-sha>` tag.
3. Remove the per-job package installation steps that the image now
   covers, keeping any step that installs job-specific extras.
4. Keep the old image references available for one release cycle in
   case a rollback is needed.

The Fedora 39 image redirects its repositories to
`archives.fedoraproject.org` inside the Dockerfile because Fedora 39 is
end-of-life and its mirrorlist no longer answers.
