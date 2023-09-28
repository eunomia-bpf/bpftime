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

#
# Setup libbpf
#
set(LIBBPF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libbpf/)
include(ExternalProject)
ExternalProject_Add(libbpf
  PREFIX libbpf
  SOURCE_DIR ${LIBBPF_DIR}/src
  CONFIGURE_COMMAND "mkdir" "-p" "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf"
  BUILD_COMMAND "INCLUDEDIR=" "LIBDIR=" "UAPIDIR=" "OBJDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf" "DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf" "make" "CFLAGS=-g -O2 -Werror -Wall -std=gnu89 -fPIC -fvisibility=hidden -DSHARED -DCUSTOM_DEFINE=1" "-j" "install"
  BUILD_IN_SOURCE TRUE
  INSTALL_COMMAND ""
  STEP_TARGETS build
  BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a
)

# Set BpfObject input parameters -- note this is usually not necessary unless
# you're in a highly vendored environment (like libbpf-bootstrap)
set(LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/libbpf)
set(LIBBPF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a)

#
add_custom_target(copy_headers ALL
  COMMENT "Copying headers"
)

function(copy_header TARGET SRC_DIR TARGET_DIR)
  file(GLOB_RECURSE FILES RELATIVE "${SRC_DIR}" "${SRC_DIR}/*")
  message(STATUS "copying ${FILES} from ${SRC_DIR} to ${TARGET_DIR}")

  foreach(file ${FILES})
    get_filename_component(PARENT_DIR "${TARGET_DIR}/${file}" DIRECTORY)
    add_custom_command(
      TARGET ${TARGET}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PARENT_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy
      ${SRC_DIR}/${file}
      ${TARGET_DIR}/${file}
      COMMENT "Copying file ${HEADER_DIRS}/${file} to ${TARGET_DIR}/${file}"
      BYPRODUCTS ${TARGET_DIR}/${file}
    )
  endforeach()
endfunction()

copy_header(copy_headers "${LIBBPF_DIR}/include/linux" "${LIBBPF_INCLUDE_DIRS}/linux")
copy_header(copy_headers "${LIBBPF_DIR}/include/uapi/linux" "${LIBBPF_INCLUDE_DIRS}/linux")

set(HEADER_FILES relo_core.h hashmap.h nlattr.h libbpf_internal.h)

foreach(file ${HEADER_FILES})
  add_custom_command(
    TARGET copy_headers
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${LIBBPF_DIR}/src/${file}
    ${LIBBPF_INCLUDE_DIRS}/bpf/${file}
    COMMENT "Copying ${file}"
    BYPRODUCTS ${LIBBPF_INCLUDE_DIRS}/bpf/${file}
  )
endforeach()

add_dependencies(copy_headers libbpf)


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

# ebpf-verifier
option(ENABLE_EBPF_VERIFIER "Whether to enable ebpf verifier" OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE YES)

if(${ENABLE_EBPF_VERIFIER})
  add_subdirectory(third_party/ebpf-verifier)
  set(EBPF_VERIFIER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ebpf-verifier/src)
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
