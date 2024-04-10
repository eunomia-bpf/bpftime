cmake_minimum_required(VERSION 3.15)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StandardSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CompilerWarnings.cmake)


add_library(bpftime_vm_compat ./src/ebpf_vm_compat.c)

target_include_directories(bpftime_vm_compat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


set_project_warnings(bpftime_vm_compat)
target_link_libraries(bpftime_vm_compat PUBLIC ${BPFTIME_BPF_RUNTIME_NAME})

add_dependencies(bpftime_vm_compat ${BPFTIME_BPF_RUNTIME_NAME})
