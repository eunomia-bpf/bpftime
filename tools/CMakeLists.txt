add_subdirectory(bpftimetool)
if(BPFTIME_ENABLE_FRIDA)
  add_subdirectory(cli)
endif()
# AOT tooling needs LLVM and libbpf; skip on QNX phase-1 where libbpf is absent
if(BPFTIME_LLVM_JIT AND BPFTIME_BUILD_WITH_LIBBPF AND NOT BPFTIME_TARGET_QNX)
  message(STATUS "Using llvm-jit")
  add_subdirectory(aot)
endif()
