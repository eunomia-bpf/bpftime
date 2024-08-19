cmake_minimum_required(VERSION 3.21)
#
# Project details
#
project(
  "vm-bpf"
  VERSION 0.1.0
  LANGUAGES C CXX
)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(NOT DEFINED SPDLOG_INCLUDE)
  message(STATUS "Adding spdlog seperately..")
  # spdlog
  add_subdirectory(../third_party/spdlog ${CMAKE_CURRENT_BINARY_DIR}/spdlog)
  set(SPDLOG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/spdlog/include)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
else()
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
endif()

if(${BPFTIME_BUILD_STANDALONE_VM})
  include(../cmake/libbpf.cmake)
  add_subdirectory(../third_party/argparse ${CMAKE_CURRENT_BINARY_DIR}/argparse)
endif()

add_subdirectory(compat)
add_subdirectory(vm-core)

#
# Unit testing setup
#
set(test_sources
  src/test.c
  src/test_jit.c
  src/test_vm.c
  src/test_core_minimal_ffi.c
)

if(BPFTIME_ENABLE_UNIT_TESTING)
  enable_testing()
  message(STATUS "Build unit tests for the project. Tests should always be found in the test folder\n")
endif()

add_subdirectory(test-runner)

