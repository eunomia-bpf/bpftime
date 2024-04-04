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

bpftime_add_library(vm-bpf
    src/ebpf_vm.cpp
    src/llvm/llvm_jit_context.cpp
    src/llvm/compiler.cpp
    src/llvm/compiler_utils.cpp
)

set_target_properties(vm-bpf PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../")

if (${LLVM_PACKAGE_VERSION} VERSION_LESS 15)
    message(FATAL_ERROR "LLVM version must be >=15")
endif()

if(ENABLE_LLVM_SHARED)
    set(LLVM_LIBS LLVM)
else()
    llvm_map_components_to_libnames(LLVM_LIBS
        Core
        OrcJIT
        Support
        nativecodegen
    )
endif()

if(NOT DEFINED SPDLOG_INCLUDE)
  message(INFO "Adding spdlo seperately..")
  # spdlog
  add_subdirectory(../../third_party/spdlog ${CMAKE_CURRENT_BINARY_DIR}/spdlog)
  set(SPDLOG_INCLUDE ../../third_party/spdlog/include)
endif()

message(STATUS "LLVM_LIBS=${LLVM_LIBS}")

target_link_libraries(vm-bpf ${LLVM_LIBS} spdlog::spdlog)
target_include_directories(vm-bpf PUBLIC ${LLVM_INCLUDE_DIRS} ${SPDLOG_INCLUDE})

bpftime_add_executable(vm-llvm-example
    example/main.cpp
)

add_dependencies(vm-llvm-example vm-bpf spdlog::spdlog)
target_link_libraries(vm-llvm-example vm-bpf)

bpftime_add_executable(vm-llvm-bpf-test
    example/main-bpf-conformance.cpp
)

add_dependencies(vm-llvm-bpf-test vm-bpf)
target_link_libraries(vm-llvm-bpf-test vm-bpf)

target_include_directories(vm-bpf PUBLIC ../include include)

if(BPFTIME_ENABLE_UNIT_TESTING)
  message(STATUS "Build unit tests for the project. Tests should always be found in the test folder\n")
  add_subdirectory(unit-test)
endif()

find_package(Boost REQUIRED)

target_include_directories(vm-bpf PRIVATE ${Boost_INCLUDE})
