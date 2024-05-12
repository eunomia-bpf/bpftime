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
# Prevent building in the source directory
#
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n \"mkdir build ; cd build ; cmake ..\" \
  \nSome junk files were created in this folder (CMakeCache.txt, CMakeFiles); you should delete those.")
endif()

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
set(CMAKE_CXX_STANDARD 20)

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



include(cmake/libbpf.cmake)

# install frida
include(cmake/frida.cmake)
include(cmake/ArchiveHelper.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE YES)

if(${BPFTIME_BUILD_STATIC_LIB}) 

endif()

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

if (${BPFTIME_BUILD_STATIC_LIB}) 
message(STATUS " Adding libraries to static archive")
bpftime_add_static_lib_component_command(runtime)
bpftime_add_static_lib_component_command(attach)
bpftime_add_static_lib_component_command(vm)
if (${ENABLE_EBPF_VERIFIER}) 
bpftime_add_static_lib_component_command(bpftime-verifier)
endif()
if(CMAKE_AR_NAME STREQUAL "libtool")
add_custom_command(OUTPUT "libbpftime.a"
  COMMAND ${CMAKE_AR} -static -o libbpftime.a ${BPFTIME_STATIC_LIB_LIBTOOL_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BPFTIME_STATIC_LIB_DEPS}
)
else()
add_custom_command(OUTPUT "libbpftime.a"
  ${BPFTIME_STATIC_LIB_AR_CMDS}
  COMMAND ${CMAKE_AR} -qcs libbpftime.a objs/*/*.o
  COMMAND ${CMAKE_COMMAND} -E remove_directory objs
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BPFTIME_STATIC_LIB_DEPS}
)
endif()

add_custom_target(bpftime_static_target ALL DEPENDS "libbpftime.a")
add_library(bpftime_static STATIC IMPORTED GLOBAL)
add_dependencies(bpftime_static bpftime_static_target)

set_target_properties(bpftime_static
PROPERTIES
IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libbpftime.a"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libbpftime.a
DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

install(TARGETS bpftime-agent bpftime_text_segment_transformer bpftime-syscall-server CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
