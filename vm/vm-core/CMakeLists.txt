cmake_minimum_required(VERSION 3.15)

add_library(bpftime_vm STATIC ./src/ebpf-vm.cpp)

target_include_directories(bpftime_vm PRIVATE ./include INTERFACE ./include)

message("The runtime is ${BPFTIME_BPF_RUNTIME_NAME}")

add_dependencies(bpftime_vm bpftime_vm_compat spdlog ${BPFTIME_BPF_RUNTIME_NAME})
target_link_libraries(bpftime_vm PUBLIC bpftime_vm_compat spdlog ${BPFTIME_BPF_RUNTIME_NAME})
