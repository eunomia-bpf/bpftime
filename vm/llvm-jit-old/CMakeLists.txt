cmake_minimum_required(VERSION 3.21)

project(
    "llvm-bpf-jit"
    LANGUAGES C CXX
    VERSION 0.1.0
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

function(bpftime_setup_target target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    target_include_directories(${target}
        PUBLIC src "include")
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

function(bpftime_add_executable target)
    add_executable(${target} ${ARGN})
    bpftime_setup_target(${target})
endfunction()

function(bpftime_add_library target)
    add_library(${target} ${ARGN})
    bpftime_setup_target(${target})
endfunction()

bpftime_add_library(bpftime_llvm_jit_vm
    src/llvm/llvm_jit_context.cpp
    src/llvm/compiler.cpp
    src/llvm/compiler_utils.cpp
    src/compat/compat_llvm.cpp
)

set_target_properties(bpftime_llvm_jit_vm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../")

find_package(LLVM REQUIRED CONFIG)

if (${LLVM_PACKAGE_VERSION} VERSION_LESS 15)
    message(FATAL_ERROR "LLVM version must be >=15")
endif()

option(ENABLE_LLVM_SHARED "Link shared library of LLVM" YES)

if(ENABLE_LLVM_SHARED)
    set(LLVM_LIBS LLVM)
else()
    llvm_map_components_to_libnames(LLVM_LIBS
        Core
        OrcJIT
        mcjit
        Support
        nativecodegen
    )
endif()

if(NOT DEFINED SPDLOG_INCLUDE)
  message(INFO "Adding spdlog seperately..")
  # spdlog
  add_subdirectory(../../third_party/spdlog ${CMAKE_CURRENT_BINARY_DIR}/spdlog)
  set(SPDLOG_INCLUDE ../../third_party/spdlog/include)
endif()

# if BPFTIME_LLVM_JIT is set, then it's built in the bpftime project.
# If not, it's built as a standalone library.
if(${BPFTIME_LLVM_JIT})
    # only build cli in the main project because it relies on libbpf
    add_subdirectory(cli)
else()
    if(${BPFTIME_ENABLE_UNIT_TESTING})
        add_subdirectory(../../third_party/Catch2 ${CMAKE_CURRENT_BINARY_DIR}/Catch2)
    endif()
    add_subdirectory(../compat ${CMAKE_CURRENT_BINARY_DIR}/compat)
endif()

message(STATUS "LLVM_LIBS=${LLVM_LIBS}")
find_package(Boost REQUIRED)

target_link_libraries(bpftime_llvm_jit_vm PUBLIC ${LLVM_LIBS} bpftime_vm_compat PRIVATE spdlog::spdlog)
    target_include_directories(bpftime_llvm_jit_vm 
    PUBLIC ${LLVM_INCLUDE_DIRS} ${SPDLOG_INCLUDE} ${Boost_INCLUDE} ../include include #LLVM jit also used these headers
)
add_dependencies(bpftime_llvm_jit_vm bpftime_vm_compat spdlog::spdlog)

if(BPFTIME_ENABLE_UNIT_TESTING)
  message(STATUS "Build unit tests for the project. Tests should always be found in the test folder\n")
  add_subdirectory(unit-test)
endif()

add_subdirectory(example)
