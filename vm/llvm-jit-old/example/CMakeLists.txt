bpftime_add_executable(vm-llvm-example
    ./main.cpp
)

add_dependencies(vm-llvm-example bpftime_llvm_jit_vm spdlog::spdlog)
target_link_libraries(vm-llvm-example bpftime_llvm_jit_vm)

