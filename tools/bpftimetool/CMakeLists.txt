add_executable(bpftimetool
  main.cpp
)
target_link_libraries(bpftimetool PRIVATE argparse)
if(${BPFTIME_ENABLE_CUDA_ATTACH})
  include(../../cmake/cuda.cmake)
  find_cuda()
  target_link_directories(bpftimetool PUBLIC ${CUDA_LIBRARY_PATH})
  target_include_directories(bpftimetool PUBLIC ${CUDA_INCLUDE_PATH})
endif()
if(${BPFTIME_BUILD_WITH_LIBBPF})
  set (LIBBPF_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/../../runtime/object/libbpf)
  target_include_directories(bpftimetool PUBLIC
    ../../third_party/
    ../../runtime/include
    ${LIBBPF_INCLUDE_DIRS}/uapi
    ${LIBBPF_INCLUDE_DIRS}
  )

  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    target_link_libraries(bpftimetool PRIVATE
      -lpthread
      -lm
      -ldl
      runtime
      bpftime-object
      ${CUDA_LIBS}
      bpftime_nv_attach_impl
    )
  else()
    target_link_libraries(bpftimetool PRIVATE
      -lpthread
      -lm
      -ldl
      runtime
      bpftime-object
    )
  endif()
else()
  target_include_directories(bpftimetool PUBLIC
    ../../third_party/
    ../../runtime/include
  )
  if(${BPFTIME_ENABLE_CUDA_ATTACH})
    target_link_libraries(bpftimetool PRIVATE
      -lpthread
      -lm
      -ldl
      runtime
      ${CUDA_LIBS}
      bpftime_nv_attach_impl
    )
  else()
    target_link_libraries(bpftimetool PRIVATE
      -lpthread
      -lm
      -ldl
      runtime
    )
  endif()
endif()

# The tool loads VMs by name through create_vm_instance, and the "llvm"
# factory registers itself only from compat_llvm.cpp's constructor. Nothing
# references that object by symbol, so the linker can drop it from the
# archive (example/attach_implementation links it whole for the same
# reason). Keep the registration unconditionally.
if(BPFTIME_LLVM_JIT)
  target_link_options(bpftimetool PRIVATE
    "-Wl,--whole-archive" "$<TARGET_FILE:bpftime_llvm_vm>" "-Wl,--no-whole-archive")
endif()

install(TARGETS bpftimetool CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
