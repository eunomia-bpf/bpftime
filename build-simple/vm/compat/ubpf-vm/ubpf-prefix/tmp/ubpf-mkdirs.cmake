# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/bpftime/bpftime/vm/compat/ubpf-vm/../../../third_party/ubpf")
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/vm/compat/ubpf-vm/../../../third_party/ubpf")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/tmp"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/src/ubpf-stamp"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/src"
  "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/src/ubpf-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/src/ubpf-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/vm/compat/ubpf-vm/ubpf-prefix/src/ubpf-stamp${cfgdir}") # cfgdir has leading slash
endif()
