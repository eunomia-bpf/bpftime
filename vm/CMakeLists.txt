cmake_minimum_required(VERSION 3.15)

#
# Project details
#
project(
  "vm-bpf"
  VERSION 0.1.0
  LANGUAGES C CXX
)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/StandardSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompilerWarnings.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(NOT DEFINED SPDLOG_INCLUDE)
  message(INFO "Adding spdlo seperately..")
  # spdlog
  add_subdirectory(../third_party/spdlog ${CMAKE_CURRENT_BINARY_DIR}/spdlog)

  # set spdlog level
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
  else()
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
  endif()

  set(SPDLOG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)
endif()

message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

if(BPFTIME_LLVM_JIT)
  message(STATUS "Using llvm-jit")
  add_subdirectory(llvm-jit)
else()
  message(STATUS "Not using llvm-jit")
  add_subdirectory(simple-jit)
endif()

set_project_warnings(vm-bpf)

# set the -static flag for static linking
if(NOT BPFTIME_ENABLE_ASAN)
  # set the -static flag for static linking
  # set_target_properties(${test_name}_Tests PROPERTIES LINK_FLAGS "-static")
  # need on qemu-user
endif()

message(STATUS "Added all header and implementation files.\n")

message(DEBUG "Successfully added all dependencies and linked against them.")

#
# Set the build/user include directories
#
message(STATUS "Finished setting up include directories.")

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
  add_subdirectory(test)
endif()

add_subdirectory(test-runner)

