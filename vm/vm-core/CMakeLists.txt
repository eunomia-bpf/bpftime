cmake_minimum_required(VERSION 3.15)

add_library(bpftime_vm STATIC ./src/ebpf-vm.cpp)

target_include_directories(bpftime_vm PRIVATE ./include INTERFACE ./include)


if(${BPFTIME_LLVM_JIT})
  target_link_options(bpftime_vm PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_llvm_vm>" "-Wl,--no-whole-archive")
  add_dependencies(bpftime_vm bpftime_vm_compat spdlog "bpftime_llvm_vm")
  target_link_libraries(bpftime_vm PUBLIC bpftime_vm_compat spdlog "bpftime_llvm_vm")
endif()

if(${BPFTIME_UBPF_JIT})
  target_link_options(bpftime_vm PUBLIC "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_ubpf_vm>" "-Wl,--no-whole-archive")
  add_dependencies(bpftime_vm bpftime_vm_compat spdlog "bpftime_ubpf_vm")
  target_link_libraries(bpftime_vm PUBLIC bpftime_vm_compat spdlog "bpftime_ubpf_vm")
endif()

