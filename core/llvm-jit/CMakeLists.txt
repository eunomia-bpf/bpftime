project(
    "llvm-bpf-jit"
    LANGUAGES C CXX
    VERSION 0.1.0
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

cmake_minimum_required(VERSION 3.15)

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

bpftime_add_library(core-bpf
    src/bpf_jit.cpp
    src/ebpf_vm.cpp
    src/bpf_jit_compile_module.cpp
)

set_target_properties(core-bpf PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../")

set(LLVM_SEARCH_PATHS
    /usr/lib/llvm-15/lib/cmake/llvm
    /usr/lib/llvm-14/lib/cmake/llvm
    /usr/lib/llvm-13/lib/cmake/llvm
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

message(STATUS "LLVM_LIBS=${LLVM_LIBS}")

target_link_libraries(core-bpf ${LLVM_LIBS})
target_include_directories(core-bpf PUBLIC ${LLVM_INCLUDE_DIRS})

bpftime_add_executable(core-llvm-example
    example/main.cpp
)

add_dependencies(core-llvm-example core-bpf)
target_link_libraries(core-llvm-example core-bpf)

bpftime_add_executable(core-llvm-bpf-test
    example/main-bpf-conformance.cpp
)

add_dependencies(core-llvm-bpf-test core-bpf)
target_link_libraries(core-llvm-bpf-test core-bpf)

target_include_directories(core-bpf PUBLIC ../include include)

# add_custom_target(core-bpf ALL)

# add_custom_command(TARGET core-bpf
#     COMMAND ar --output "${CMAKE_CURRENT_BINARY_DIR}/extracted-objs/core-bpf" -x $<TARGET_FILE:core-bpf>
#     BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/extracted-objs/core-bpf
# )

# set(RAW_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

# # set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
# foreach(lib ${LLVM_LIBS})
#     message(STATUS "Finding llvm lib ${lib}")
#     find_library("local_${lib}" NAMES ${lib} PATHS ${LLVM_LIBRARY_DIRS} REQUIRED)
#     set(lib_path "${local_${lib}}")
#     message(STATUS "Found ${lib}: ${lib_path}")
#     add_custom_command(TARGET core-bpf
#         COMMAND ar --output "${CMAKE_CURRENT_BINARY_DIR}/extracted-objs/${lib}" -x ${lib_path}
#         BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/extracted-objs/${lib}
#     )
# endforeach()

# set(CMAKE_FIND_LIBRARY_SUFFIXES ${RAW_CMAKE_FIND_LIBRARY_SUFFIXES})
