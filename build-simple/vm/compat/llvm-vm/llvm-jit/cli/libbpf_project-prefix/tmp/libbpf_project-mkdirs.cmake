# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/bpftime/bpftime/vm/llvm-jit/cli/libbpf_project")
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/vm/llvm-jit/cli/libbpf_project")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src/libbpf_project-build"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/tmp"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src/libbpf_project-stamp"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src/libbpf_project-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src/libbpf_project-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/llvm-vm/llvm-jit/cli/libbpf_project-prefix/src/libbpf_project-stamp${cfgdir}") # cfgdir has leading slash
endif()
