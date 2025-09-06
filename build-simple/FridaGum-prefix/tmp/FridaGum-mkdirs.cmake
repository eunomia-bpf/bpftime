# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum")
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum-build"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/tmp"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum-stamp"
  "/home/runner/work/bpftime/bpftime/third_party/frida"
  "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/bpftime/bpftime/build-simple/FridaGum-prefix/src/FridaGum-stamp${cfgdir}") # cfgdir has leading slash
endif()
