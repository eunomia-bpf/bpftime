
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



if(${BPFTIME_LLVM_JIT})
  message(STATUS "Using llvm-jit")
  add_subdirectory(llvm-jit)
  set(BPFTIME_BPF_RUNTIME_NAME "bpftime_llvm_jit")
else()
  message(STATUS "Not using llvm-jit")
  add_subdirectory(simple-jit)
  set(BPFTIME_BPF_RUNTIME_NAME "bpftime_simple_jit")
endif()

add_subdirectory(compat)

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
