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

include(cmake/libbpf.cmake)

# install frida
include(cmake/frida.cmake)

# Define a helper function
function(add_ebpf_program_target target_name source_file output_file)
  # opensnoop.bpf.o
  execute_process(COMMAND bash -c "uname -m | sed 's/x86_64/x86/' \
| sed 's/arm.*/arm/' \
| sed 's/aarch64/arm64/' \
| sed 's/ppc64le/powerpc/' \
| sed 's/mips.*/mips/' \
| sed 's/riscv64/riscv/' \
| sed 's/loongarch64/loongarch/'"
    OUTPUT_VARIABLE UNAME_ARCH
    COMMAND_ERROR_IS_FATAL ANY
  )
  string(STRIP ${UNAME_ARCH} UNAME_ARCH_STRIPPED)
  add_custom_target(${target_name} ALL
    COMMAND clang -O2 -target bpf -c -g -D__TARGET_ARCH_${UNAME_ARCH_STRIPPED} -I${CMAKE_SOURCE_DIR}/third_party/vmlinux/${UNAME_ARCH_STRIPPED} -I${LIBBPF_INCLUDE_DIRS}/uapi -I${LIBBPF_INCLUDE_DIRS} ${source_file} -o ${output_file}
    BYPRODUCTS ${output_file}
    SOURCES ${source_file}
  )
  add_dependencies(${target_name} copy_headers)
endfunction()

set(CMAKE_POSITION_INDEPENDENT_CODE YES)

# ebpf-verifier
option(ENABLE_EBPF_VERIFIER "Whether to enable ebpf verifier" OFF)

if(${ENABLE_EBPF_VERIFIER})
  add_subdirectory(bpftime-verifier)
endif()

# spdlog
add_subdirectory(third_party/spdlog)

set(SPDLOG_INCLUDE {CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)

# main library
add_subdirectory(vm)
add_subdirectory(runtime)
add_subdirectory(tools)

# benchmark that requires bpftime libraries
add_subdirectory(benchmark)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

install(TARGETS bpftime-agent bpftime-agent-transformer bpftime-syscall-server CONFIGURATIONS Release Debug DESTINATION ~/.bpftime)
