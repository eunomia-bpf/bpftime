cmake_minimum_required(VERSION 3.15)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer")

#
# Project details
#
project(
  "bpftime"
  VERSION 0.1.0
  LANGUAGES C CXX
)

#
# Set project options
#
include(cmake/CompilerWarnings.cmake)
include(cmake/StandardSettings.cmake)
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_program(LLVM_CONFIG "llvm-config")
execute_process(COMMAND ${LLVM_CONFIG} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${LLVM_INCLUDE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling ubsan for Debug builds; Processor=${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm((32.*)|(v6.*)|(v7.*))")
    message(STATUS "Linking libatomic on arm32 machines..")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic")
  endif()
endif()

message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

if(UNIX)
  add_compile_options("$<$<CONFIG:DEBUG>:-D_DEBUG>") # this will allow to use same _DEBUG macro available in both Linux as well as Windows - MSCV environment. Easy to put Debug specific code.
endif(UNIX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

#
# Setup alternative names
#
if(${PROJECT_NAME}_USE_ALT_NAMES)
  string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
  string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPERCASE)
else()
  set(PROJECT_NAME_LOWERCASE ${PROJECT_NAME})
  set(PROJECT_NAME_UPPERCASE ${PROJECT_NAME})
endif()

#
# Prevent building in the source directory
#
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n")
endif()

include(cmake/libbpf.cmake)

# install frida
include(cmake/frida.cmake)

# Set default value for BPFTIME_LLVM_JIT to ON
option(BPFTIME_LLVM_JIT "Use LLVM as jit backend." ON)

set(CMAKE_POSITION_INDEPENDENT_CODE YES)

if(${ENABLE_EBPF_VERIFIER})
  add_subdirectory(bpftime-verifier)
else()
  message(STATUS "Skipping ebpf verifier")
  # Catch2
  add_subdirectory(third_party/Catch2)
endif()

# spdlog
add_subdirectory(third_party/spdlog)

set(SPDLOG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)

# argparse
add_subdirectory(third_party/argparse)

# main library
add_subdirectory(vm)
add_subdirectory(vm/cli)

add_subdirectory(attach)

add_subdirectory(runtime)
if (${BUILD_BPFTIME_DAEMON})
  add_subdirectory(daemon)
endif()
add_subdirectory(tools)

if(${BUILD_ATTACH_IMPL_EXAMPLE})
  add_subdirectory(example/attach_implementation)
endif()

# benchmark that requires bpftime libraries
add_subdirectory(benchmark)


set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

install(TARGETS bpftime-agent bpftime_text_segment_transformer bpftime-syscall-server CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
