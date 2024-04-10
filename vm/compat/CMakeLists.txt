cmake_minimum_required(VERSION 3.15)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StandardSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CompilerWarnings.cmake)


add_library(bpftime_vm_compat INTERFACE)

target_include_directories(bpftime_vm_compat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )


target_link_libraries(bpftime_vm_compat INTERFACE ${BPFTIME_BPF_RUNTIME_NAME})

add_dependencies(bpftime_vm_compat ${BPFTIME_BPF_RUNTIME_NAME})
