add_subdirectory(bpftimetool)
add_subdirectory(cli)
# AOT tooling needs LLVM; skip on QNX phase-1 when LLVM JIT is off
if(BPFTIME_LLVM_JIT AND NOT BPFTIME_TARGET_QNX)
  message(STATUS "Using llvm-jit")
  add_subdirectory(aot)
endif()
