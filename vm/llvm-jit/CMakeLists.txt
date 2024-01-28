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

set(LLVM_SEARCH_PATHS
    /usr/lib/llvm-15/lib/cmake/llvm
    /usr/lib/llvm-14/lib/cmake/llvm
    /usr/lib/llvm-13/lib/cmake/llvm
    /usr/lib64/llvm15/lib/cmake/llvm
)

find_package(LLVM REQUIRED CONFIG
    PATHS ${LLVM_SEARCH_PATHS}
    NO_DEFAULT_PATH)

llvm_map_components_to_libnames(LLVM_LIBS
    Core
    OrcJIT
    Support
    nativecodegen
)

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

include(FetchContent)
FetchContent_Declare(
    picosha2
    GIT_REPOSITORY https://github.com/okdshin/PicoSHA2
    GIT_TAG 27fcf69
)
FetchContent_MakeAvailable(picosha2)
FetchContent_GetProperties(picosha2)

# get_target_property(picosha2_INCLUDE picosha2 INTERFACE_INCLUDE_DIRECTORIES)
set(picosha2_INCLUDE ${picosha2_SOURCE_DIR})
target_include_directories(vm-bpf PRIVATE ${picosha2_INCLUDE})
add_dependencies(vm-bpf picosha2)

find_package(Boost REQUIRED)

target_include_directories(vm-bpf PRIVATE ${Boost_INCLUDE})
