add_library(bpftime_llvm_vm_obj OBJECT compat_llvm.cpp)

add_dependencies(bpftime_llvm_vm_obj llvmbpf_vm spdlog::spdlog bpftime_vm_compat)

# Object sources need the same definitions as the archive's consumers. In
# particular, mixing header-only spdlog here with the agent's compiled spdlog
# creates separate loggers, bypassing the agent's configured output sink.
target_link_libraries(bpftime_llvm_vm_obj PRIVATE llvmbpf_vm spdlog::spdlog)

add_library(bpftime_llvm_vm STATIC $<TARGET_OBJECTS:bpftime_llvm_vm_obj>)
target_link_libraries(bpftime_llvm_vm PUBLIC bpftime_vm_compat spdlog::spdlog llvmbpf_vm)

target_include_directories(bpftime_llvm_vm_obj PRIVATE ../../llvm-jit/include ${CMAKE_CURRENT_SOURCE_DIR} ${SPDLOG_INCLUDE} ../include)

if(BPFTIME_BUILD_WITH_LIBBPF)
  add_subdirectory(../../llvm-jit ${CMAKE_CURRENT_BINARY_DIR}/llvm-jit)
else()
  add_subdirectory(../../llvm-jit ${CMAKE_CURRENT_BINARY_DIR}/llvm-jit EXCLUDE_FROM_ALL)
endif()
