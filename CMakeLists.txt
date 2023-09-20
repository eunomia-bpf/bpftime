cmake_minimum_required(VERSION 3.15)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling ubsan for Debug builds; Processor=${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")

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

# install frida
include(cmake/frida.cmake)

# ebpf-verifier
option(ENABLE_EBPF_VERIFIER "Whether to enable ebpf verifier" OFF)

if(${ENABLE_EBPF_VERIFIER})
  add_subdirectory(third_party/ebpf-verifier)
endif()

# main library
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(tools)

# For Windows, it is necessary to link with the MultiThreaded library.
# Depending on how the rest of the project's dependencies are linked, it might be necessary
# to change the line to statically link with the library.
#
# This is done as follows:
#
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#
# On Linux and Mac this variable is ignored. If any issues rise from it, try commenting it out
# and letting CMake decide how to link with it.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
