# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore")
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore-build"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/tmp"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore-stamp"
  "/home/runner/work/bpftime/bpftime/third_party/frida"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaCore-prefix/src/FridaCore-stamp${cfgdir}") # cfgdir has leading slash
endif()
