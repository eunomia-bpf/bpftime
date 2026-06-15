# bpftime release surface plan

This document tracks the install-channel work needed to claim official bpftime
names without publishing empty or misleading packages.

## Package names to claim

| Channel | Name | Status / next step |
| --- | --- | --- |
| GitHub | `eunomia-bpf/bpftime` | Exists; keep releases canonical. |
| GHCR | `ghcr.io/eunomia-bpf/bpftime` | Exists; Docker workflow now prepares `latest`, `vX.Y.Z`, `X.Y.Z`, and sha tags. |
| Homebrew tap | `eunomia-bpf/tap/bpftime` | Prepare after release assets exist or source-build formula is reliable. |
| Homebrew core | `bpftime` | Submit after tap formula passes and source/release artifact story is clean. |
| APT/PPA | `bpftime`, `libbpftime-dev` | Needs `.deb` packaging, signing key, and Launchpad/self-hosted repo access. |
| Fedora/COPR | `bpftime`, `bpftime-devel` | Needs RPM spec/COPR config and account access. |
| crates.io | `bpftime`, `bpftime-sys` | Publish only with real Rust API/FFI or CLI wrapper. |
| PyPI | `bpftime` | Publish only with real Python binding or CLI wrapper. |
| npm | `bpftime`, `@eunomia-bpf/bpftime` | Publish only with real Node/WASM/native addon story. |
| vcpkg | `bpftime` | High priority for C/C++ users after install targets are stable. |
| ConanCenter | `bpftime/<version>` | High priority for C/C++ users after install targets are stable. |
| nixpkgs | `bpftime` | Submit after reproducible build expression is available. |
| conda-forge | `bpftime` | Submit after dependencies and release assets are stable. |

## Current blockers

- The `v0.2.0` GitHub release currently has no downloadable release assets.
- The source build depends on submodules and a large native dependency stack; the
  Homebrew core, Debian, Fedora, vcpkg, and conda-forge submissions need a stable
  source-build recipe.
- CMake currently installs runtime artifacts under `~/.bpftime`; distro packages
  may need explicit install layout decisions for `/usr/bin`, `/usr/lib`, and
  `/usr/include`.

## Immediate TODO

1. Build and upload release assets for Linux amd64:
   - `bpftime-linux-amd64.tar.gz`
   - checksums
   - optional SBOM/provenance
2. Define package split:
   - `bpftime`: CLI/runtime tools
   - `libbpftime`: shared runtime libraries if applicable
   - `libbpftime-dev` / `bpftime-devel`: headers, CMake/pkg-config metadata
3. Create official Homebrew tap formula after assets or source-build recipe are
   reliable.
4. Create `.deb` and `.rpm` packaging from the same install layout.
5. Start upstream submissions in parallel:
   - Homebrew core PR
   - Debian WNPP/ITP
   - Ubuntu needs-packaging
   - Fedora package review
   - vcpkg port
   - ConanCenter recipe
   - nixpkgs PR
   - conda-forge feedstock

## Accounts/secrets needed

- GitHub package write access is already covered by `GITHUB_TOKEN` for GHCR.
- Homebrew tap requires `eunomia-bpf/homebrew-tap` access.
- APT/self-hosted repo requires a GPG signing key and `apt.eunomia.dev` hosting.
- Launchpad PPA requires Launchpad account/team and GPG key.
- COPR/Fedora official requires Fedora/COPR account path.
- vcpkg, ConanCenter, nixpkgs, and conda-forge require GitHub PR review cycles.
