add_library(bpftime_llvm_vm STATIC compat_llvm.cpp)

add_dependencies(bpftime_llvm_vm llvmbpf_vm spdlog::spdlog bpftime_vm_compat)

target_link_libraries(bpftime_llvm_vm PUBLIC bpftime_vm_compat spdlog::spdlog llvmbpf_vm)

target_include_directories(bpftime_llvm_vm PRIVATE ../../llvm-jit/include ${CMAKE_CURRENT_SOURCE_DIR} ${SPDLOG_INCLUDE})

add_subdirectory(../../llvm-jit ${CMAKE_CURRENT_BINARY_DIR}/llvm-jit)

