add_subdirectory(bpftimetool)
add_subdirectory(cli)
# TODO: remove this
# if(BPFTIME_LLVM_JIT)
  message(STATUS "Using llvm-jit")
  add_subdirectory(aot)
# endif()
