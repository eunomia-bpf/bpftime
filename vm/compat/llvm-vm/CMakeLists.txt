add_library(bpftime_llvm_vm_obj OBJECT compat_llvm.cpp)

add_dependencies(bpftime_llvm_vm_obj llvmbpf_vm spdlog::spdlog bpftime_vm_compat)

add_library(bpftime_llvm_vm STATIC $<TARGET_OBJECTS:bpftime_llvm_vm_obj>)
target_link_libraries(bpftime_llvm_vm PUBLIC bpftime_vm_compat spdlog::spdlog llvmbpf_vm)

target_include_directories(bpftime_llvm_vm_obj PRIVATE ../../llvm-jit/include ${CMAKE_CURRENT_SOURCE_DIR} ${SPDLOG_INCLUDE} ../include)

add_subdirectory(../../llvm-jit ${CMAKE_CURRENT_BINARY_DIR}/llvm-jit)

