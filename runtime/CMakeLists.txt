cmake_minimum_required(VERSION 3.15)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

#
# Project details
#
project(
  "runtime"
  VERSION 0.1.0
  LANGUAGES C CXX
)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/StandardSettings.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

if(UNIX)
  add_compile_options("$<$<CONFIG:DEBUG>:-D_DEBUG>") # this will allow to use same _DEBUG macro available in both Linux as well as Windows - MSCV environment. Easy to put Debug specific code.
endif(UNIX)

#
# Prevent building in the source directory
#
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n")
endif()

#
# Create library, setup header and source files
#
find_package(Boost REQUIRED)

# Find all headers and implementation files
message(STATUS "Building for architecture: ${ARCH}")

set(sources
  src/attach/bpf_attach_ctx.cpp

  src/handler/handler_manager.cpp
  src/handler/map_handler.cpp
  src/handler/perf_event_handler.cpp
  src/handler/prog_handler.cpp
  src/handler/epoll_handler.cpp

  src/bpftime_shm.cpp
  src/bpftime_shm_internal.cpp
  src/bpftime_shm_json.cpp
  src/bpftime_prog.cpp
  src/ufunc.cpp
  src/bpf_helper.cpp

  src/bpf_map/userspace/array_map.cpp
  src/bpf_map/userspace/hash_map.cpp
  src/bpf_map/userspace/ringbuf_map.cpp
  src/bpf_map/userspace/perf_event_array_map.cpp
  src/bpf_map/userspace/per_cpu_array_map.cpp
  src/bpf_map/userspace/per_cpu_hash_map.cpp
  src/bpf_map/userspace/prog_array.cpp
  extension/extension_helper.cpp
)

if(BPFTIME_BUILD_KERNEL_BPF)
  list(APPEND sources
    src/bpf_map/shared/array_map_kernel_user.cpp
    src/bpf_map/shared/hash_map_kernel_user.cpp
    src/bpf_map/shared/percpu_array_map_kernel_user.cpp
    src/bpf_map/shared/perf_event_array_kernel_user.cpp
  )
endif()


set(headers
  include/
)
message(INFO " Headers: ${headers}")

message(INFO " Found the following sources: ${sources}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
else()
  add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
endif()

add_library(
  ${PROJECT_NAME}
  ${sources}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h
  COMMAND /bin/bash ${CMAKE_CURRENT_SOURCE_DIR}/generate_syscall_id_table.sh "${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h"
  USES_TERMINAL
)
add_custom_target(
  syscall_id_table
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/syscall_id_list.h
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../vm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
  ${SPDLOG_INCLUDE}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  bpftime_vm_compat
  spdlog::spdlog
  bpftime_base_attach_impl
)
add_dependencies(${PROJECT_NAME} bpftime_vm_compat FridaGum spdlog::spdlog libbpf bpftime_base_attach_impl)

if(BPFTIME_ENABLE_IOURING_EXT)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
    uring
  )
endif()

if(${ENABLE_EBPF_VERIFIER})
  target_include_directories(${PROJECT_NAME} PRIVATE ${BPFTIME_VERIFIER_INCLUDE})
  target_link_libraries(${PROJECT_NAME} PRIVATE bpftime-verifier)
  add_dependencies(${PROJECT_NAME} bpftime-verifier)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_EBPF_VERIFIER ENABLE_BPFTIME_VERIFIER)
endif()

message(DEBUG "Found the following sources: ${sources}")

message(DEBUG "Found the following headers: ${headers}")

# set the -static flag for static linking
if(NOT BPFTIME_ENABLE_ASAN)
  # set the -static flag for static linking
  # set_target_properties(${test_name}_Tests PROPERTIES LINK_FLAGS "-static")
  # need on qemu-user
endif()

message(STATUS "Added all header and implementation files.\n")

#
# Set the project standard and warnings
#
set_project_warnings(runtime)

message(DEBUG "Applied compiler warnings. Using standard ${CMAKE_CXX_STANDARD}.")

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  ${LIBBPF_LIBRARIES}
  ${FRIDA_GUM_INSTALL_DIR}/libfrida-gum.a
  -lpthread
  -lm
  -ldl
  -lz
  -lelf
  bpftime_base_attach_impl
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${LIBBPF_INCLUDE_DIRS}/uapi
  ${LIBBPF_INCLUDE_DIRS}
  ${FRIDA_GUM_INSTALL_DIR}
  $<INSTALL_INTERFACE:runtime>
  $<INSTALL_INTERFACE:runtime/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

message(DEBUG "Successfully added all dependencies and linked against them.")

set(BPFTIME_RUNTIME_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(object)
add_subdirectory(agent)
add_subdirectory(syscall-server)

#
# Unit testing setup
#
if(BPFTIME_ENABLE_UNIT_TESTING)
  enable_testing()
  message(STATUS "Build unit tests for the project. Tests should always be found in the test folder\n")
  add_subdirectory(test)
  add_subdirectory(unit-test)
endif()
