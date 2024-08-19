cmake_minimum_required(VERSION 3.15)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StandardSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CompilerWarnings.cmake)


add_library(bpftime_vm_compat INTERFACE)

target_include_directories(bpftime_vm_compat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${SPDLOG_INCLUDE})

if(${BPFTIME_LLVM_JIT})
  message(STATUS "Using llvm-jit")
  add_subdirectory(llvm-vm)
  set(BPFTIME_BPF_RUNTIME_NAME "bpftime_llvm_jit_vm")
else()
  message(STATUS "Using ubpf-jit")
  add_subdirectory(ubpf-vm)
  set(BPFTIME_BPF_RUNTIME_NAME "bpftime_ubpf_vm")
endif()