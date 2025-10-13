---
name: Package bpftime for APT
about: Request to package bpftime for Debian/Ubuntu APT repositories
title: '[Feature] Package bpftime for APT distribution'
labels: 'enhancement, packaging'
assignees: ''

---

## Summary

Package bpftime for distribution via APT package manager for Debian and Ubuntu systems.

## Motivation

Currently, bpftime requires manual installation from source or using Docker images. Packaging bpftime for APT would:
- Simplify installation for Debian/Ubuntu users
- Enable automatic dependency resolution
- Facilitate easier updates and version management
- Improve accessibility for users who prefer system package managers
- Support better integration with CI/CD pipelines

## Current Installation Methods

As documented in `installation.md`, users currently need to:
1. Manually install dependencies via apt-get
2. Clone the repository and initialize submodules
3. Build from source using cmake
4. Manually install to `~/.bpftime`

Or use Docker images from GitHub packages.

## Proposed Solution

Create Debian packages (.deb) for bpftime and its components:

### Package Structure

**Main Package: `bpftime`**
- CLI tool for injecting agent & server
- Installed to `/usr/bin/bpftime`

**Additional Packages:**
- `bpftime-vm` - VM and compiler tools
- `bpftime-daemon` - Daemon for kernel interaction
- `bpftime-dev` - Development headers and libraries
- `libbpftime-agent` - Shared libraries for agent functionality
- `libbpftime-syscall-server` - Syscall server library

### Dependencies to Package

Key dependencies that need to be specified:
- libelf1, libelf-dev
- zlib1g-dev
- libboost-all-dev (or libboost1.74-all-dev)
- binutils-dev
- libyaml-cpp-dev
- clang-17, llvm-17, llvm-17-dev (or compatible versions)
- cmake, make, git (build dependencies)

### Build Configuration

The package should build with optimized settings similar to:
```bash
cmake -Bbuild \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBPFTIME_LLVM_JIT=1 \
    -DBUILD_BPFTIME_DAEMON=1
```

### Installation Paths

- Binaries: `/usr/bin/`
- Libraries: `/usr/lib/x86_64-linux-gnu/` or `/usr/lib/`
- Headers: `/usr/include/bpftime/`
- Documentation: `/usr/share/doc/bpftime/`
- Examples: `/usr/share/doc/bpftime/examples/`

## Implementation Steps

1. Create `debian/` directory with packaging metadata:
   - `control` - Package metadata and dependencies
   - `rules` - Build instructions
   - `changelog` - Version history
   - `copyright` - License information
   - `compat` - Debhelper compatibility level
   - `install` - File installation rules

2. Create packaging scripts:
   - Pre/post installation scripts
   - Package removal scripts
   - Maintainer scripts for library registration

3. Set up PPA (Personal Package Archive) for distribution:
   - Configure Launchpad or other PPA hosting
   - Set up automated builds for different Ubuntu versions
   - Configure GPG signing for packages

4. Create CI/CD pipeline for package building:
   - Automated package building on new releases
   - Testing installation on clean systems
   - Validation of package integrity

5. Documentation:
   - Update `installation.md` with apt installation instructions
   - Add package maintenance documentation
   - Create guide for package maintainers

## Benefits

- **Ease of Installation**: `sudo apt install bpftime`
- **Dependency Management**: Automatic handling of all dependencies
- **Version Control**: Easy updates via `apt upgrade`
- **System Integration**: Proper integration with system libraries and paths
- **Wider Adoption**: Lower barrier to entry for new users
- **CI/CD Friendly**: Easier to integrate into automated workflows

## Alternatives Considered

1. **Snap Package**: Alternative package format for Ubuntu
   - Pros: Cross-distribution, automatic updates
   - Cons: Larger size, some performance overhead

2. **AppImage**: Portable application format
   - Pros: No installation required, portable
   - Cons: No dependency management, larger size

3. **Flatpak**: Another universal package format
   - Pros: Sandboxing, cross-distribution
   - Cons: Overhead, complexity for system-level tools

APT packaging is preferred as it's the native package manager for Debian/Ubuntu systems and best suited for system-level development tools.

## Additional Context

**Current project state:**
- License: MIT (package-friendly)
- Active development with stable releases
- Documentation available
- Build system using CMake (standard for packaging)
- Dependencies clearly documented in Dockerfile

**References:**
- Debian packaging guidelines: https://www.debian.org/doc/manuals/maint-guide/
- Ubuntu packaging guide: https://packaging.ubuntu.com/html/
- Similar projects packaged for APT: bpftrace, libbpf-dev, bcc-tools

**Related Files:**
- `Dockerfile` - Shows dependency installation and build process
- `installation.md` - Current installation documentation
- `CMakeLists.txt` - Build configuration
- `Makefile` - Build automation

## Checklist

- [ ] Create debian packaging files
- [ ] Test package building on Ubuntu 20.04, 22.04, 24.04
- [ ] Test package building on Debian 11, 12
- [ ] Set up PPA for distribution
- [ ] Configure automated builds
- [ ] Update documentation
- [ ] Announce availability to community
